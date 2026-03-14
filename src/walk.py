import os, time, numpy as np, pinocchio, crocoddyl, signal, sys
import yaml
from pinocchio.robot_wrapper import RobotWrapper
from crocoddyl import MeshcatDisplay
from utils.custom_biped import SimpleBipedGaitProblem, plotSolution
from utils.analysis_util import save_and_plot_robot_data
from convert_to_mimickit import convert_solvers_to_pkl

# -------------------- Runtime flags -----------------------------------------
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT    = "plot"    in sys.argv or "CROCODDYL_PLOT"    in os.environ
WITHSAVE    = "save"    in sys.argv or "CROCODDYL_SAVE"    in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# -------------------- Paths --------------------------------------------------
TIMESTEP  = 0.01
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../config/robot_configs.yaml")

with open(CONFIG_PATH, "r") as f:
    robots = yaml.safe_load(f)

robot_name = sys.argv[1] if len(sys.argv) > 1 else "tocabi"
robot_cfg  = robots[robot_name]

rel_urdf = robot_cfg["urdf"]
rel_mesh = robot_cfg["mesh"]

URDF_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", rel_urdf))
MESH_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", rel_mesh))

# -------------------- Robot --------------------------------------------------
robot = RobotWrapper.BuildFromURDF(
    URDF_PATH,
    [MESH_DIR],
    pinocchio.JointModelFreeFlyer()
)

model = robot.model
model.effortLimit *= 1.0

# Use foot frame names from YAML
left_foot_name  = robot_cfg["left_foot"]
right_foot_name = robot_cfg["right_foot"]

print("========== URDF INFO ==========")
print("nq (configuration dimension):", model.nq)
print("nv (velocity dimension):", model.nv)
print("nu (actuated DoFs):", model.nv - 6)

# -------------------- Initial state -----------------------------------------
q0_list = robot_cfg.get("q0", None)
if q0_list is not None:
    q0 = np.array(q0_list, dtype=float)
    assert q0.size == model.nq, f"q0 size {q0.size} != model.nq {model.nq}"
else:
    print("[WARN] q0 not found in YAML, using pinocchio.neutral(model).")
    q0 = pinocchio.neutral(model)

data = model.createData()
pinocchio.forwardKinematics(model, data, q0)
pinocchio.updateFramePlacements(model, data)

z_RF = data.oMf[model.getFrameId(right_foot_name)].translation[2]
z_LF = data.oMf[model.getFrameId(left_foot_name)].translation[2]
q0[2] -= 0.5 * (z_RF + z_LF)
q0[2] +=0.1585

# Store reference configuration
model.referenceConfigurations["half_sitting"] = q0.copy()

v0 = np.zeros(model.nv)
x0 = np.concatenate([q0, v0])

# -------------------- Gait generator ----------------------------------------
rightFoot, leftFoot = right_foot_name, left_foot_name
gait = SimpleBipedGaitProblem(model, rightFoot, leftFoot)

# -------------------- Parameter setting --------------------------------------

GAITPHASES = [
    dict(
        walking=dict(
            stepX        = 0.2,
            stepY        = 0.0,
            stepYaw      = 0.0,
            stepHeight   = 0.15,
            timeStep     = TIMESTEP,
            stepKnots    = 60,  # * TIMESTEP = SSP Duration
            supportKnots = 20,  # * TIMESTEP = DSP Duration
        )
    )
    for _ in range(3)
]

# ----------------------------------------------------------
# Solve BoxFDDP
solver = []  # Stores the DDP solver (ddp) for each phase for later use
for phase in GAITPHASES:
    cfg = phase["walking"]

    # Create walking problem
    pb = gait.createWalkingProblem(
        x0,
        cfg["stepX"],
        cfg["stepY"],
        cfg["stepYaw"],
        cfg["stepHeight"],
        cfg["timeStep"],
        cfg["stepKnots"],
        cfg["supportKnots"],
    )

    # Create Box-constrained DDP solver
    ddp = crocoddyl.SolverBoxFDDP(pb)
    ddp.th_stop = 1e-12

    callbacks = [crocoddyl.CallbackVerbose()]
    if WITHPLOT:
        callbacks.append(crocoddyl.CallbackLogger())
    ddp.setCallbacks(callbacks)

    # Initial guess for state trajectory
    xs = [x0] * (pb.T + 1)

    # Initial guess for control trajectory: quasi-static controls
    us = ddp.problem.quasiStatic(xs[:-1])

    # Solve optimal control problem
    ddp.solve(xs, us, 100, False, 0.1)
                                             
    solver.append(ddp)                                      # Stores the solved ddp solver in the list for later use.
    
    # x0 = ddp.xs[-1]                                         # Updates x0 to be the final state of the current phase. 
                                                            # This becomes the starting point for the next phase (to ensure continuity between motions).
    q_final = ddp.xs[-1][:model.nq]
    x0 = np.concatenate([q_final, np.zeros(model.nv)])

convert_solvers_to_pkl(
    solver_list = solver,
    model       = model,
    output_path = "data/motions/tocabi_walk.pkl",
    timestep    = TIMESTEP,
)

# ----------------------------------------------------------
# Visualization
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

# ----------------------------------------------------------
# Plot
if WITHPLOT or WITHSAVE:
    print(f"\n[Analysis] Processing data for {robot_name}...")
    
    xs_all = []
    us_all = []
    
    for i, ddp in enumerate(solver):
        # 마지막 phase가 아니면 중복 방지를 위해 마지막 x를 제외하고 합침
        if i < len(solver) - 1:
            xs_all.extend(list(ddp.xs)[:-1])
        else:
            xs_all.extend(list(ddp.xs))
        
        # u(토크) 데이터 정리
        for u in ddp.us:
            if u.size == 0: # 혹시 제어량이 없는 노드가 있다면 0으로 채움
                us_all.append(np.zeros(model.nv - 6))
            else:
                us_all.append(u)

    # 리스트를 numpy array로 변환
    xs_all = np.array(xs_all)
    us_all = np.array(us_all)

    # 기존 backflip 코드에서 썼던 함수 그대로 호출
    # 이 함수가 내부적으로 관절 12개(혹은 그 이상)의 q, v, u를 각각 plot하고 저장합니다.
    save_and_plot_robot_data(
        model=model,
        xs=xs_all,
        us=us_all,
        dt=TIMESTEP,
        robot_name=robot_name
    )
# if WITHPLOT:
#     plotSolution(solver, bounds=False, figIndex=1, show=False)
#     for i, ddp in enumerate(solver):
#         log = ddp.getCallbacks()[1]
#         crocoddyl.plotConvergence(
#             log.costs, log.pregs, log.dregs, log.grads,
#             log.stops, log.steps,
#             figTitle=f"walking (phase {i})", figIndex=i + 3,
#             show=(i == len(solver) - 1),
#         )

#----------------------------------------------------------
# Save
# if WITHSAVE:
#     SAVE_DIR = "./walking_dataset"
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     all_q = []   # joint positions (full q)
#     all_v = []   # joint velocities (full v)
#     all_u = []   # torques (actuated)

#     nu = model.nv - 6       
#     nv = model.nv
#     nq = model.nq

#     for ddp in solver:
#         T = len(ddp.us)      

#         for k in range(T):
#             x = ddp.xs[k]           
#             u = ddp.us[k]         

#             q = x[:nq].copy()
#             v = x[nq:].copy()
            
#             if u.size == 0:
#                 u_full = np.zeros(nu)
#             elif u.size == nu:
#                 u_full = u.copy()
#             else:
#                 raise RuntimeError(f"Unexpected control size: {u.size}, expected {nu}")
            
#             all_q.append(q)
#             all_v.append(v)
#             all_u.append(u_full)

#     all_q = np.array(all_q)
#     all_v = np.array(all_v)
#     all_u = np.array(all_u)

#     np.savetxt(os.path.join(SAVE_DIR, "q_traj.txt"), all_q, fmt="%.6f")
#     np.savetxt(os.path.join(SAVE_DIR, "v_traj.txt"), all_v, fmt="%.6f")
#     np.savetxt(os.path.join(SAVE_DIR, "u_traj.txt"), all_u, fmt="%.6f")

#     print(f"[SAVE] Saved q, v, u trajectories to {SAVE_DIR}")