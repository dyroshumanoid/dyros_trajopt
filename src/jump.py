import os, time, numpy as np, pinocchio, crocoddyl, signal, sys
import yaml
from pinocchio.robot_wrapper import RobotWrapper
from utils.custom_biped import SimpleBipedGaitProblem, plotSolution
# from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution
from crocoddyl import MeshcatDisplay

# -------------------- Runtime flags -----------------------------------------
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT    = "plot"    in sys.argv or "CROCODDYL_PLOT"    in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# -------------------- Paths --------------------------------------------------
TIMESTEP  = 0.05
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
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
z0   = 0.5 * (z_RF + z_LF)
q0[2] -= z0

model.referenceConfigurations["half_sitting"] = q0.copy()

v0 = np.zeros(model.nv)
x0 = np.concatenate([q0, v0])

# ----------------------------------------------------------
# 4) Gait (walking + jumping) problem
rightFoot, leftFoot = right_foot_name, left_foot_name
gait = SimpleBipedGaitProblem(model, rightFoot, leftFoot, fwddyn=True)

# -------------------- Parameter setting --------------------------------------
GAITPHASES = [
    {
        "walking": {
            "stepX":        0.3,
            "stepY":        0.0,
            "stepYaw":      0.0,
            "stepHeight":   0.1,
            "timeStep":     TIMESTEP,
            "stepKnots":    15,
            "supportKnots": 5,
        }
    },
    {
        "jumping": {
            "jumpHeight":   0.1,
            "jumpLength":   [0.3, 0.0, 0.0],
            "timeStep":     TIMESTEP,
            "groundKnots":  9,
            "flyingKnots":  6,
        }
    },
    {
        "walking": {
            "stepX":        0.3,
            "stepY":        0.0,
            "stepYaw":      0.0,
            "stepHeight":   0.1,
            "timeStep":     TIMESTEP,
            "stepKnots":    15,
            "supportKnots": 5,
        }
    },
]

solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            problem = gait.createWalkingProblem(
                x0,
                value["stepX"],
                value["stepY"],
                value["stepYaw"],
                value["stepHeight"],
                value["timeStep"],
                value["stepKnots"],
                value["supportKnots"],
            )
        elif key == "jumping":
            problem = gait.createJumpingProblem(
                x0,
                value["jumpHeight"],
                value["jumpLength"],
                value["timeStep"],
                value["groundKnots"],
                value["flyingKnots"],
            )
        else:
            raise ValueError(f"Unknown phase key: {key}")

        solver[i] = crocoddyl.SolverIntro(problem)
        solver[i].th_stop = 1e-7

    print(f"*** SOLVE {key} ***")
    if WITHPLOT:
        solver[i].setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    solver[i].solve(xs, us, 100, False)

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]

# ----------------------------------------------------------
# 5) Meshcat visualization
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
        time.sleep(1.0)

# ----------------------------------------------------------
# 6) Logs & plots
if WITHPLOT:
    plotSolution(solver, bounds=False, figIndex=1, show=False)
    for i, ddp in enumerate(solver):
        log = ddp.getCallbacks()[1]
        crocoddyl.plotConvergence(
            log.costs, log.pregs, log.dregs, log.grads,
            log.stops, log.steps,
            figTitle=f"phase {i} ({list(GAITPHASES[i].keys())[0]})",
            figIndex=i + 3,
            show=(i == len(solver) - 1),
        )
