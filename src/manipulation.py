import os
import signal
import sys
import time

import numpy as np
import pinocchio
import crocoddyl
import yaml

from pinocchio.robot_wrapper import RobotWrapper
from utils.custom_biped import plotSolution

# -------------------- Runtime flags -----------------------------------------
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT    = "plot"    in sys.argv or "CROCODDYL_PLOT"    in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# -------------------- Paths --------------------------------------------------
TIMESTEP  = 0.01
T         = 500  # number of knots

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

print("URDF:", URDF_PATH)
print("Mesh:", MESH_DIR)

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
left_hand_name  = robot_cfg["left_hand"]
right_hand_name = robot_cfg["right_hand"]

print("========== URDF INFO ==========")
print("nq (configuration dimension):", model.nq)
print("nv (velocity dimension):", model.nv)
print("nu (actuated DoFs):", model.nv - 6)

# -------------------- State / actuation -------------------------------------
state     = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

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

# Adjust base height so both feet touch the ground (z = 0)
z_RF = data.oMf[model.getFrameId(right_foot_name)].translation[2]
z_LF = data.oMf[model.getFrameId(left_foot_name)].translation[2]
q0[2] -= 0.5 * (z_RF + z_LF)

# Reference configuration (same 스타일 as walk.py)
model.referenceConfigurations["half_sitting"] = q0.copy()

v0 = np.zeros(model.nv)
x0 = np.concatenate([q0, v0])

# Recompute kinematics with shifted base
pinocchio.forwardKinematics(model, data, q0)
pinocchio.updateFramePlacements(model, data)

# -------------------- Frames and initial poses ------------------------------
rightFoot   = right_foot_name
leftFoot    = left_foot_name
leftHand    = left_hand_name
rightHand   = right_hand_name

rightFootId = model.getFrameId(rightFoot)
leftFootId  = model.getFrameId(leftFoot)
leftHandId  = model.getFrameId(leftHand)
rightHandId = model.getFrameId(rightHand)

rfPos0 = data.oMf[rightFootId].translation.copy()
lfPos0 = data.oMf[leftFootId].translation.copy()
leftStart  = data.oMf[leftHandId].translation.copy()
rightStart = data.oMf[rightHandId].translation.copy()

# CoM reference (between feet in xy, actual COM height in z)
comRef = (rfPos0 + lfPos0) / 2.0
comRef[2] = pinocchio.centerOfMass(model, data, q0)[2].item()

# -------------------- Hand trajectory (task space) --------------------------
leftGoal  = np.array([+0.5, +0.4, +1.5])
rightGoal = np.array([+0.5, -0.4, +0.9])

leftTraj  = [(1.0 - t / T) * leftStart  + (t / T) * leftGoal  for t in range(T)]
rightTraj = [(1.0 - t / T) * rightStart + (t / T) * rightGoal for t in range(T)]

# -------------------- Contact model (feet fixed on ground) ------------------
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel.addContact(
    "left_contact",
    crocoddyl.ContactModel6D(
        state,
        leftFootId,
        pinocchio.SE3.Identity(),
        pinocchio.LOCAL,
        actuation.nu,
        np.array([0.0, 0.0]),
    ),
)
contactModel.addContact(
    "right_contact",
    crocoddyl.ContactModel6D(
        state,
        rightFootId,
        pinocchio.SE3.Identity(),
        pinocchio.LOCAL,
        actuation.nu,
        np.array([0.0, 0.0]),
    ),
)

# -------------------- Joint limits (state bounds) ---------------------------
maxfloat = sys.float_info.max
xlb = np.concatenate(
    [
        -maxfloat * np.ones(7),
        model.lowerPositionLimit[8:],   # actual joint lower limits
        -maxfloat * np.ones(state.nv),
    ]
)
xub = np.concatenate(
    [
        maxfloat * np.ones(7),
        model.upperPositionLimit[8:],   # actual joint upper limits
        maxfloat * np.ones(state.nv),
    ]
)
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
xLimitResidual   = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
limitCost        = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

# -------------------- State / control regularization ------------------------
# Keep upper body / joints relatively stiff (similar idea as walking weights)
stateWeights  = np.array(
    [0.0] * 3               # base position x,y,z
    + [10.0] * 3            # base orientation
    + [100.0] * 12          # leg joints (for example, stronger regularization)
    + [0.01] * (state.nv - 18)
    + [10.0] * state.nv
)
xActivation  = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
xTActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)

xResidual    = crocoddyl.ResidualModelState(state, x0, actuation.nu)
uResidual    = crocoddyl.ResidualModelControl(state, actuation.nu)

xRegCost     = crocoddyl.CostModelResidual(state, xActivation,  xResidual)
xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)
uRegCost     = crocoddyl.CostModelResidual(state, uResidual)

# -------------------- CoM cost (optional, currently weight = 0) -------------
comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
comTrack    = crocoddyl.CostModelResidual(state, comResidual)

# -------------------- Running models (time-varying hand targets) ------------
runningModels = []
for t in range(T):
    # Left hand target
    target_L   = leftTraj[t]
    placementL = pinocchio.SE3(np.eye(3), target_L)
    residual_L = crocoddyl.ResidualModelFramePlacement(
        state, leftHandId, placementL, actuation.nu
    )
    act_L      = crocoddyl.ActivationModelWeightedQuad(
        (np.array([1.0] * 3 + [1e-4] * 3)) ** 2
    )
    cost_L     = crocoddyl.CostModelResidual(state, act_L, residual_L)

    # Right hand target
    target_R   = rightTraj[t]
    placementR = pinocchio.SE3(np.eye(3), target_R)
    residual_R = crocoddyl.ResidualModelFramePlacement(
        state, rightHandId, placementR, actuation.nu
    )
    act_R      = crocoddyl.ActivationModelWeightedQuad(
        (np.array([1.0] * 3 + [1e-4] * 3)) ** 2
    )
    cost_R     = crocoddyl.CostModelResidual(state, act_R, residual_R)

    # Cost sum
    costModel = crocoddyl.CostModelSum(state, actuation.nu)
    costModel.addCost("track_L",  cost_L,        1e2)
    costModel.addCost("track_R",  cost_R,        1e2)
    costModel.addCost("xReg",     xRegCost,      1e-3)
    costModel.addCost("uReg",     uRegCost,      1e-4)
    costModel.addCost("limit",    limitCost,     1e3)
    costModel.addCost("comTrack", comTrack,      0.0)

    dmodel       = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel, costModel
    )
    runningModel = crocoddyl.IntegratedActionModelEuler(dmodel, TIMESTEP)
    runningModels.append(runningModel)

# -------------------- Terminal model (final hand pose targets) --------------
placementL_T = pinocchio.SE3(np.eye(3), leftGoal)
placementR_T = pinocchio.SE3(np.eye(3), rightGoal)

residual_L_T = crocoddyl.ResidualModelFramePlacement(
    state, leftHandId, placementL_T, actuation.nu
)
residual_R_T = crocoddyl.ResidualModelFramePlacement(
    state, rightHandId, placementR_T, actuation.nu
)

act_L_T = crocoddyl.ActivationModelWeightedQuad(
    (np.array([1.0] * 3 + [1e-4] * 3)) ** 2
)
act_R_T = crocoddyl.ActivationModelWeightedQuad(
    (np.array([1.0] * 3 + [1e-4] * 3)) ** 2
)

cost_L_T = crocoddyl.CostModelResidual(state, act_L_T, residual_L_T)
cost_R_T = crocoddyl.CostModelResidual(state, act_R_T, residual_R_T)

terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel.addCost("track_L",  cost_L_T,      1e2)
terminalCostModel.addCost("track_R",  cost_R_T,      1e2)
terminalCostModel.addCost("xReg",     xRegTermCost,  1e-3)
terminalCostModel.addCost("limit",    limitCost,     1e3)
terminalCostModel.addCost("comTrack", comTrack,      0.0)

dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel, terminalCostModel
)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0.0)

# -------------------- Shooting problem & solver -----------------------------
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
solver  = crocoddyl.SolverBoxFDDP(problem)
solver.th_stop = 1e-12

if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [x0] * (T + 1)
us = solver.problem.quasiStatic([x0] * T)
solver.solve(xs, us, 500, False, 1e-12)

# -------------------- Final EE positions ------------------------------------
xT = solver.xs[-1]
pinocchio.forwardKinematics(model, data, xT[: state.nq])
pinocchio.updateFramePlacements(model, data)

finalLeft  = data.oMf[leftHandId].translation
finalRight = data.oMf[rightHandId].translation

print("Left hand reached  = ({:.3f}, {:.3f}, {:.3f})".format(*finalLeft))
print("Right hand reached = ({:.3f}, {:.3f}, {:.3f})".format(*finalRight))

# -------------------- Visualization (Meshcat, walk.py 스타일) ---------------
from crocoddyl import MeshcatDisplay

if WITHDISPLAY:
    display = MeshcatDisplay(robot)
    display.forceScale = 1.0
    display.frictionConeScale = 0.07
    display.withContactForces = True
    display.withFrictionCones = True
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(TIMESTEP)

# -------------------- Plotting ----------------------------------------------
if WITHPLOT:
    log = solver.getCallbacks()[1]
    plotSolution(solver, bounds=False, figIndex=1, show=False)

    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        figIndex=3,
    )
