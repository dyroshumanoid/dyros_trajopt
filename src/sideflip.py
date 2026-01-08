#!/usr/bin/env python3
import os
import sys
import time
import signal
import yaml
import numpy as np
import pinocchio
import crocoddyl

from pinocchio.robot_wrapper import RobotWrapper
from crocoddyl import MeshcatDisplay
from scipy.spatial.transform import Rotation as R

from utils.sideflip_util import SideflipProblem
from utils.analysis_util import save_and_plot_robot_data
def smoothstep(a):
    return 3*a*a - 2*a*a*a

def smoothstep_d(a):
    return 6*a - 6*a*a
def unwrap_roll_pair(roll0, rollf):
    # roll0를 음수 쪽으로 보내고, rollf가 roll0보다 더 작게(더 많이 회전) 되도록
    if roll0 > 0:
        roll0 -= 2.0 * np.pi
    while rollf > roll0:
        rollf -= 2.0 * np.pi
    return roll0, rollf

def hermite_roll(roll0, rollf, w0, wf, t, T):
    s = t / T
    s2 = s*s
    s3 = s2*s

    h00 =  2*s3 - 3*s2 + 1
    h10 =      s3 - 2*s2 + s
    h01 = -2*s3 + 3*s2
    h11 =      s3 - s2

    roll = h00*roll0 + h10*(T*w0) + h01*rollf + h11*(T*wf)

    dh00 = 6*s2 - 6*s
    dh10 = 3*s2 - 4*s + 1
    dh01 = -dh00
    dh11 = 3*s2 - 2*s

    w = (dh00*roll0 + dh10*(T*w0) + dh01*rollf + dh11*(T*wf)) / T
    return roll, w
def make_stage1_guess_sideflip(
    sideflip: SideflipProblem,
    x0: np.ndarray,
    jump_height: float,
    jump_length: list,
    dt: float,
    num_ground_knots: int,
    num_flying_knots: int,
    v_liftoff: float,
    T: int, 
):
    # ---- takeoff target state 만들기 (util과 동일) ----
    x_takeoff = sideflip.x_takeoff.copy()
    v_roll_kick = -11.0
    x_takeoff[sideflip.robot_model.nq : sideflip.robot_model.nq + 6] = np.array(
        [0.0, 0.0, v_liftoff, v_roll_kick, 0.0, 0.0]
    )

    # ---- takeoff guess ----
    xs_takeoff = sideflip.state_interp(x0, x_takeoff, num_ground_knots)  # (N, nx)

    # ---- flyup posture guess: takeoff -> flying -> inverse (util과 동일) ----
    n_mid = int(num_flying_knots * 0.5)
    n_end = num_flying_knots - n_mid
    traj_a = sideflip.state_interp(x_takeoff, sideflip.x_flying, n_mid)
    traj_b = sideflip.state_interp(sideflip.x_flying, sideflip.x_inverse, n_end)
    xs_flyup = np.vstack([traj_a, traj_b])  # (num_flying_knots, nx)

    # ---- flyup base position guess ----
    base_pos_ref_0 = x_takeoff[:3].copy()
    base_pos_ref_peak = base_pos_ref_0 + np.array(
        [jump_length[0] / 2.0, jump_length[1] / 2.0, jump_height]
    )
    g = 9.81
    v0z = x_takeoff[sideflip.robot_model.nq + 2]

    base_pos_ref_traj = []
    for k in range(num_flying_knots):
        t = (k + 1) * dt
        alpha = (k + 1) / num_flying_knots

        xy = base_pos_ref_0[0:2] + (base_pos_ref_peak[0:2] - base_pos_ref_0[0:2]) * alpha
        z_ref = x_takeoff[2] + v0z * t - 0.5 * g * t * t
        z_peak = x_takeoff[2] + jump_height
        z = min(z_ref, z_peak)

        base_pos_ref_traj.append([xy[0], xy[1], z])

    xs_flyup[:, :3] = np.array(base_pos_ref_traj)

    # ---- flyup roll + quat + wx guess (util과 동일: Hermite) ----
    yaw_s, pitch_s, roll_s = R.from_quat(x_takeoff[3:7]).as_euler("zyx")
    if roll_s > 0:
        roll_s -= 2.0 * np.pi

    roll_f = float(sideflip.flyup_roll_target)
    Tf = num_flying_knots * dt
    w0_roll = float(x_takeoff[sideflip.robot_model.nq + 3])
    w_end_roll = (roll_f - roll_s) / Tf

    base_roll_ref_traj = []
    base_roll_w_ref = []
    for k in range(num_flying_knots):
        t = (k + 1) * dt
        roll, wroll = sideflip.hermite_roll(roll_s, roll_f, w0_roll, w_end_roll, t, Tf)
        base_roll_ref_traj.append(roll)
        base_roll_w_ref.append(wroll)

    base_quat_ref_traj = []
    for k in range(num_flying_knots):
        yaw_k, pitch_k, _ = R.from_quat(xs_flyup[k, 3:7]).as_euler("zyx")
        q = R.from_euler("zyx", [yaw_k, pitch_k, base_roll_ref_traj[k]]).as_quat()
        base_quat_ref_traj.append(q)

    xs_flyup[:, 3:7] = np.array(base_quat_ref_traj)
    xs_flyup[:, sideflip.robot_model.nq + 3] = np.array(base_roll_w_ref)
    xs_flyup[:, sideflip.robot_model.nq + 4] = 0.0
    xs_flyup[:, sideflip.robot_model.nq + 5] = 0.0

    # ---- concat ----
    xs_all = np.vstack([xs_takeoff, xs_flyup])  # (K, nx)  K가 T인지 T+1인지 모름

    xs_init = [xs_all[k].copy() for k in range(xs_all.shape[0])]

    target_len = T + 1

    while len(xs_init) < target_len:
        xs_init.append(xs_init[-1].copy())
    while len(xs_init) > target_len:
        xs_init.pop()


    return xs_init
def make_stage2_guess_sideflip(x0, x_land, T, dt, model, roll_target=None):
    nq = model.nq
    xs = [x0.copy() for _ in range(T + 1)]

    p0 = x0[0:3].copy()
    pf = x_land[0:3].copy()


    ypr0 = R.from_quat(x0[3:7]).as_euler("zyx")
    yprf = R.from_quat(x_land[3:7]).as_euler("zyx")

    roll0 = ypr0[2]
    rollf = float(roll_target) if roll_target is not None else yprf[2]

    roll0, rollf = unwrap_roll_pair(roll0, rollf)

    w0_roll = float(x0[nq + 3])   
    w_end_roll = 0.0              
    Tf = T * dt

    g = 9.81
    v0z = x0[nq + 2]

    for k in range(T + 1):
        a = k / T
        s = smoothstep(a)
        sd = smoothstep_d(a) / (T * dt)  # ds/dt

        # ---------- position guess ----------
        x = p0[0] + (pf[0] - p0[0]) * s
        y = p0[1] + (pf[1] - p0[1]) * s

        t = k * dt
        z_ball = p0[2] + v0z * t - 0.5 * g * t * t
        z = max(z_ball, pf[2])

        # ---------- yaw/pitch keep ----------
        yaw   = ypr0[0] + (yprf[0] - ypr0[0]) * s
        pitch = ypr0[1] + (yprf[1] - ypr0[1]) * s

        # ---------- roll Hermite (roll + wx) ----------
        roll, wx = hermite_roll(roll0, rollf, w0_roll, w_end_roll, t, Tf)

        q = R.from_euler("zyx", [yaw, pitch, roll]).as_quat()

        # velocities (init guess용)
        vx = (pf[0] - p0[0]) * sd
        vy = (pf[1] - p0[1]) * sd
        vz = v0z - g * t

        wy = 0.0
        wz = 0.0

        xk = xs[k]
        xk[0:3] = np.array([x, y, z])
        xk[3:7] = q
        xk[nq:nq+3] = np.array([vx, vy, vz])
        xk[nq+3:nq+6] = np.array([wx, wy, wz])
        xs[k] = xk

    # quat sign continuity
    for k in range(1, T + 1):
        if np.dot(xs[k-1][3:7], xs[k][3:7]) < 0:
            xs[k][3:7] *= -1.0

    # 마지막은 target 정확히
    xs[-1][:] = x_land[:]
    return xs


# -------------------- Runtime flags -----------------------------------------
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT    = "plot"    in sys.argv or "CROCODDYL_PLOT"    in os.environ
WITHSAVE    = "save"    in sys.argv or "CROCODDYL_SAVE"    in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# -------------------- Paths --------------------------------------------------
TIMESTEP   = 0.01
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../config/robot_configs.yaml")
WEIGHT_PATH = os.path.join(BASE_DIR, "../config/robot_weights.yaml")

# -------------------- Load robot config -------------------------------------
with open(CONFIG_PATH, "r") as f:
    robots = yaml.safe_load(f)

if os.path.isfile(WEIGHT_PATH):
    with open(WEIGHT_PATH, "r") as f:
        all_weights = yaml.safe_load(f)
    print(f"[sideflip] Loaded robot weights from: {WEIGHT_PATH}")
else:
    all_weights = {}
    print(f"[sideflip][WARN] Weight file not found at {WEIGHT_PATH}, using defaults")

# default: simple_humanoid
robot_name = sys.argv[1] if len(sys.argv) > 1 else "simple_humanoid"
if robot_name not in robots:
    raise RuntimeError(f"[sideflip] Unknown robot '{robot_name}' in {CONFIG_PATH}")

robot_cfg = robots[robot_name]

rel_urdf = robot_cfg["urdf"]
rel_mesh = robot_cfg["mesh"]

URDF_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", rel_urdf))
MESH_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", rel_mesh))

if not os.path.isfile(URDF_PATH):
    raise FileNotFoundError(f"[sideflip] URDF not found: {URDF_PATH}")

# -------------------- Build robot model -------------------------------------
robot = RobotWrapper.BuildFromURDF(
    URDF_PATH,
    [MESH_DIR],
    pinocchio.JointModelFreeFlyer()
)
model = robot.model

# Try to load reference configurations from SRDF if it exists
srdf_path = URDF_PATH.replace(".urdf", ".srdf")
if os.path.isfile(srdf_path):
    try:
        # Python binding for pinocchio::srdf::loadReferenceConfigurations
        pinocchio.loadReferenceConfigurations(model, srdf_path)
        print(f"[sideflip] Loaded reference configurations from: {srdf_path}")
    except Exception as e:
        print(f"[sideflip] Failed to load SRDF ({srdf_path}): {e}")
else:
    print(f"[sideflip] SRDF file not found, skip: {srdf_path}")

print("========== URDF INFO ==========")
print("nq (configuration dimension):", model.nq)
print("nv (velocity dimension):", model.nv)
print("nu (actuated DoFs):", model.nv - 6)

# -------------------- Initial state (x0) ------------------------------------
if "side_sitting" in model.referenceConfigurations:
    q0 = model.referenceConfigurations["side_sitting"].copy()
    print("[sideflip] q0 from SRDF: side_sitting")
elif "half_sitting" in model.referenceConfigurations:
    q0 = model.referenceConfigurations["half_sitting"].copy()
    print("[sideflip] q0 fallback: half_sitting")
# Fallback: neutral
else:
    print("[sideflip][WARN] q0 not found, using pinocchio.neutral(model)")
    q0 = pinocchio.neutral(model)

v0 = np.zeros(model.nv)
x0 = np.concatenate([q0, v0])

# -------------------- Sideflip problem setup --------------------------------
base_frame_name   = robot_cfg.get("torso", "base_link")
left_foot_name    = robot_cfg["left_foot"]
right_foot_name   = robot_cfg["right_foot"]

current_robot_weights = all_weights.get(robot_name, {})

print("========== Sideflip Frames ==========")
print("Base frame   :", base_frame_name)
print("Left  foot   :", left_foot_name)
print("Right foot   :", right_foot_name)

sideflip = SideflipProblem(
    robot_model=model,
    base_frame_name=base_frame_name,
    rf_contact_frame_name=right_foot_name,
    lf_contact_frame_name=left_foot_name,
    integrator="rk4",
    control="zero",
    weights=current_robot_weights,
    flyup_roll_target=-np.pi,
    land_roll_target=-np.pi*2      
)

# ============================================================================
# 2. STATE BOUNDS (x bounds) - matched to code2 style
# ============================================================================

# Base position bounds (x, y, z)
pos_lb = -10.0 * np.ones(3)
pos_ub =  10.0 * np.ones(3)

# Base orientation bounds (quaternion) - loose but keeps unit-ish
quat_lb = -np.ones(4)
quat_ub =  np.ones(4)

# Base linear velocity bounds
lin_vel_lb = -30.0 * np.ones(3)
lin_vel_ub =  30.0 * np.ones(3)

# Base angular velocity bounds
ang_vel_lb = -np.pi * 2 * 20 * np.ones(3)
ang_vel_ub =  np.pi * 2 * 20 * np.ones(3)

# Joint position bounds (skip the free-flyer: first 7 dofs)
joint_pos_lb = model.lowerPositionLimit[7:]
joint_pos_ub = model.upperPositionLimit[7:]

# Joint velocity bounds (nv - 6 joints, excluding base)
joint_vel_limit = np.pi * 2 * 10 * np.ones(model.nv - 6)
joint_vel_lb = -joint_vel_limit
joint_vel_ub =  joint_vel_limit

# # Joint torque bounds
# model.effortLimit *= 10.0

# Concatenate into full state bounds [pos, quat, joint_pos, lin_vel, ang_vel, joint_vel]
x_lb = np.concatenate(
    [pos_lb, quat_lb, joint_pos_lb, lin_vel_lb, ang_vel_lb, joint_vel_lb]
)
x_ub = np.concatenate(
    [pos_ub, quat_ub, joint_pos_ub, lin_vel_ub, ang_vel_ub, joint_vel_ub]
)

nx = sideflip.state.nx
assert x_lb.size == nx, f"[sideflip] x_lb size {x_lb.size} != nx {nx}"
assert x_ub.size == nx, f"[sideflip] x_ub size {x_ub.size} != nx {nx}"

# ============================================================================
# 3. PHASE PARAMETERS (SIDEFLIP_STAGES) - matched to code2 style
# ============================================================================

g = 9.81
jump_height = 0.3  # [m] apex height (same as backflip)
T = np.sqrt(2.0 * jump_height / g)
num_flying_knots = int((2.0 * T / TIMESTEP - 1) / 2)
v_liftoff = np.sqrt(2.0 * g * jump_height)

# You can tune jump_length, knot numbers exactly as in code2
SIDEFLIP_STAGES = [
    dict(
        name="first_stage",
        kind="first",
        jump_height=jump_height,
        jump_length=[0.0, 0.1, 0.0],    # move backward along +y
        dt=TIMESTEP,
        num_ground_knots=30,
        num_flying_knots=num_flying_knots,
        v_liftoff=v_liftoff,
    ),
    
    dict(
        name="second_stage",
        kind="second",
        jump_length=[0.0, 0.2, 0.0],    # continue backward travel
        dt=TIMESTEP,
        num_ground_knots=65,
        num_flying_knots=num_flying_knots,
    ),
]

# -------------------- Solve sideflip stages ---------------------------------
x_traj = []   # list of all states across stages
u_traj = []   # list of all controls across stages
solver = []   # store ddp solver for each stage (for display)

for stage in SIDEFLIP_STAGES:
    print(f"\n========== SOLVE {stage['name']} ==========")

    # 1) build problem 먼저
    if stage["kind"] == "first":
        problem = sideflip.create_sideflip_problem_first_stage(
            x0=x0,
            jump_height=stage["jump_height"],
            jump_length=stage["jump_length"],
            dt=stage["dt"],
            num_ground_knots=stage["num_ground_knots"],
            num_flying_knots=stage["num_flying_knots"],
            v_liftoff=stage["v_liftoff"],
            x_lb=x_lb,
            x_ub=x_ub,
            
        )
    else:
        problem = sideflip.create_sideflip_problem_second_stage(
            x0=x0,
            jump_length=stage["jump_length"],
            dt=stage["dt"],
            num_ground_knots=stage["num_ground_knots"],
            num_flying_knots=stage["num_flying_knots"],
            x_lb=x_lb,
            x_ub=x_ub,
        )
    
    # 2) solver
    ddp = crocoddyl.SolverFDDP(problem)
    ddp.th_stop = 1e-7
    callbacks = [crocoddyl.CallbackVerbose()]
    if WITHPLOT:
        callbacks.append(crocoddyl.CallbackLogger())
    ddp.setCallbacks(callbacks)

    # 3) warm start (xs_init)
    if stage["kind"] == "first":
        xs_init = make_stage1_guess_sideflip(
            sideflip=sideflip,
            x0=x0,
            jump_height=stage["jump_height"],
            jump_length=stage["jump_length"],
            dt=stage["dt"],
            num_ground_knots=stage["num_ground_knots"],
            num_flying_knots=stage["num_flying_knots"],
            v_liftoff=stage["v_liftoff"],
            T=problem.T, 
        )
    else:
        x_land_target = sideflip.x_landing.copy()
        x_land_target[0] += stage["jump_length"][0]
        x_land_target[1] += stage["jump_length"][1]
        x_land_target[2] += stage["jump_length"][2]

        yaw, pitch, _ = R.from_quat(x_land_target[3:7]).as_euler("zyx")
        x_land_target[3:7] = R.from_euler(
            "zyx", [yaw, pitch, sideflip.land_roll_target]
        ).as_quat()

        xs_init = make_stage2_guess_sideflip(
            x0=x0,
            x_land=x_land_target,
            T=problem.T,
            dt=stage["dt"],
            model=model,
            roll_target=sideflip.land_roll_target,
        )
    # 4) warm start (us_init)
    assert len(xs_init) == problem.T + 1
    us_qs = problem.quasiStatic(xs_init[:-1])
    us_init = [np.array(u).copy() for u in us_qs]

    # 5) solve
    ddp.solve(xs_init, us_init, 1000, False)

    # stage chaining
    x0 = ddp.xs[-1].copy()

    solver.append(ddp)
    x_traj.extend(ddp.xs)
    u_traj.extend(ddp.us)

# -------------------- Convert trajectories to 2D arrays ----------------------
nq = model.nq
nx = sideflip.state.nx
nu = model.nv - 6  # actuated joints (excluding floating base)

# States
x_list = []
for x in x_traj:
    x_arr = np.asarray(x).reshape(-1)
    if x_arr.size != nx:
        raise RuntimeError(f"[sideflip] Unexpected state size: {x_arr.size}, expected {nx}")
    x_list.append(x_arr)
x_traj = np.vstack(x_list) if x_list else np.zeros((0, nx))

# Controls (pad zeros if nu = 0 at some knots)
u_list = []
for u in u_traj:
    u_arr = np.asarray(u).reshape(-1)

    if u_arr.size == 0:
        u_arr = np.zeros(nu)
    elif u_arr.size != nu:
        raise RuntimeError(
            f"[sideflip] Unexpected control size: {u_arr.size}, expected {nu}"
        )

    u_list.append(u_arr)
u_traj = np.vstack(u_list) if u_list else np.zeros((0, nu))

print("\n========== Sideflip optimization done ==========")
print("Total knots:", x_traj.shape[0])
print("x_traj shape:", x_traj.shape)
print("u_traj shape:", u_traj.shape)


# ---------------------------------------------------------------
if WITHPLOT:
    print(f"\n[Analysis] Plotting data for {robot_name}...")
    save_and_plot_robot_data(
        model=model,
        xs=x_traj,
        us=u_traj,
        dt=TIMESTEP,
        robot_name=robot_name
    )


# -------------------- Optional: save trajectory -----------------------------
if WITHSAVE:
    SAVE_DIR = os.path.join(BASE_DIR, "../sideflip_dataset")
    os.makedirs(SAVE_DIR, exist_ok=True)

    np.savetxt(os.path.join(SAVE_DIR, "x_traj.txt"), x_traj)
    np.savetxt(os.path.join(SAVE_DIR, "u_traj.txt"), u_traj)
    print(f"[sideflip] Saved trajectories to: {SAVE_DIR}")

# -------------------- Display with Meshcat ----------------------------------
if WITHDISPLAY:
    display = MeshcatDisplay(robot)
    display.forceScale = 1.0
    display.frictionConeScale = 0.07
    display.withContactForces = True
    display.withFrictionCones = True
    display.rate = -1
    display.freq = 1
    while True:
        for ddp in solver:
            display.displayFromSolver(ddp)
        time.sleep(2.0)

# If no display, just exit
