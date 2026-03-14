#!/usr/bin/env python3
import os
import sys
import time
import signal
import yaml
import numpy as np
import pinocchio
import crocoddyl
import matplotlib.pyplot as plt


from pinocchio.robot_wrapper import RobotWrapper
from crocoddyl import MeshcatDisplay
from scipy.spatial.transform import Rotation as R

from utils.sideflip_util import SideflipProblem
from utils.analysis_util import save_and_plot_robot_data

# -------------------- Runtime flags -----------------------------------------
WITHDISPLAY = True
WITHPLOT    = False
WITHSAVE    = True
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
jump_height = 0.4  # [m] apex height tocabi 0.4
T = np.sqrt(2.0 * jump_height / g)
num_flying_knots = int((2.0 * T / TIMESTEP - 1) / 2)
v_liftoff = np.sqrt(2.0 * g * jump_height)
v_roll_kick = (-np.pi*2) / T

SIDEFLIP_STAGES = [
    dict(
        name="first_stage",
        kind="first",
        jump_height=jump_height,
        jump_length=[0.0, 0.09, 0.0],    # move sideward along +y
        dt=TIMESTEP,
        num_ground_knots=50,
        num_flying_knots=num_flying_knots,
        v_liftoff=v_liftoff,
        v_roll_kick=v_roll_kick,
    ),
    
    dict(
        name="second_stage",
        kind="second",
        jump_length=[0.0, 0.18, 0.0],    # continue sideward travel
        dt=TIMESTEP,
        num_ground_knots=55, #58
        num_flying_knots=num_flying_knots,
    ),
]

# -------------------- Solve sideflip stages ---------------------------------
x_traj = []   # list of all states across stages
u_traj = []   # list of all controls across stages
solver = []   # store ddp solver for each stage (for display)

for stage in SIDEFLIP_STAGES:
    print(f"\n========== SOLVE {stage['name']} ==========")

    if stage["kind"] == "first":
        problem = sideflip.create_sideflip_problem_first_stage(
            x0=x0,
            jump_height=stage["jump_height"],
            jump_length=stage["jump_length"],
            dt=stage["dt"],
            num_ground_knots=stage["num_ground_knots"],
            num_flying_knots=stage["num_flying_knots"],
            v_liftoff=stage["v_liftoff"],
            v_roll_kick=stage["v_roll_kick"],
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
    
    ddp = crocoddyl.SolverFDDP(problem)
    ddp.th_stop = 1e-7
    callbacks = [crocoddyl.CallbackVerbose()]
    if WITHPLOT:
        callbacks.append(crocoddyl.CallbackLogger())
    ddp.setCallbacks(callbacks)

    xs_init = [x0.copy() for _ in range(problem.T + 1)]
    us_qs = problem.quasiStatic(xs_init[:-1])
    us_init = [np.array(u).copy() for u in us_qs]

    ddp.solve(xs_init, us_init, 1000, False)

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


if WITHPLOT:
    print(f"\n[Analysis] Plotting data for {robot_name}...")
    save_and_plot_robot_data(
        model=model,
        xs=x_traj,
        us=u_traj,
        dt=TIMESTEP,
        robot_name=robot_name,
    )
    # plt.pause(0.1) 

# --------------------Display with Meshcat -------------------------
if WITHDISPLAY:
    display = MeshcatDisplay(robot)
    display.forceScale = 1.0
    display.withContactForces = True
    
    print("\n[Display] Meshcat display is starting...")
    try:
        while True:
            for ddp_solver in solver:
                display.displayFromSolver(ddp_solver)
            
            plt.pause(0.01) 
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping display...")


# -------------------- Optional: save trajectory -----------------------------
# if WITHSAVE:
#     SAVE_DIR = os.path.join(BASE_DIR, "../sideflip_dataset")
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     np.savetxt(os.path.join(SAVE_DIR, "x_traj.txt"), x_traj)
#     np.savetxt(os.path.join(SAVE_DIR, "u_traj.txt"), u_traj)
#     print(f"[sideflip] Saved trajectories to: {SAVE_DIR}")

# # If no display, just exit
