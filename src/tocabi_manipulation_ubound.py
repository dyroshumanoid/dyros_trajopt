import os
import signal
import sys
import time

import numpy as np
import pinocchio

import crocoddyl
from pinocchio.robot_wrapper import RobotWrapper
from utils.custom_biped import plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Load robot
urdf_path = "/home/kwan/humanoid-trajopt-playground/robots/tocabi/robots/dyros_tocabi.urdf"
mesh_dir  = "/home/kwan/humanoid-trajopt-playground/robots/tocabi/meshes"
assert os.path.isfile(urdf_path), "Check URDF PATH"

robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir], pinocchio.JointModelFreeFlyer())
model = robot.model
model.effortLimit *= 1.0

# Create data structures
data = model.createData()
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Set integration time
T = 400
TIMESTEP = 0.01

# Initialize reference state and CoM
q0 = np.array([0,0,0,  0,0,0,1,
               0,0,-0.24, 0.6,-0.36,0,
               0,0,-0.24, 0.6,-0.36,0,
               0,0,0,
               0.3,0.3,1.5,-1.27,-1,0,-1,0,
               0,0,
               -0.3,-0.3,-1.5,1.27,1,0,1,0])
assert q0.size == model.nq

rightFoot = "R_Foot_Link"
leftFoot = "L_Foot_Link"
leftHand = "L_Wrist2_Link"
rightHand = "R_Wrist2_Link"
rightFootId = model.getFrameId(rightFoot)
leftFootId = model.getFrameId(leftFoot)
leftHandId = model.getFrameId(leftHand)
rightHandId = model.getFrameId(rightHand)

pinocchio.forwardKinematics(model, data, q0)
pinocchio.updateFramePlacements(model, data)
rfPos0 = data.oMf[rightFootId].translation
lfPos0 = data.oMf[leftFootId].translation
z0 = 0.5 * (rfPos0[2] + lfPos0[2])
q0[2] -= z0
x0 = np.concatenate([q0, np.zeros(model.nv)])

pinocchio.forwardKinematics(model, data, q0)
pinocchio.updateFramePlacements(model, data)
leftStart = data.oMf[leftHandId].translation
rightStart = data.oMf[rightHandId].translation
comRef = (rfPos0 + lfPos0) / 2
comRef[2] = pinocchio.centerOfMass(model, data, q0)[2].item()

leftGoal  = np.array([+0.5, +0.4, +1.5])
rightGoal = np.array([+0.5, -0.4, +0.9])
leftTraj = [(1 - t / T) * leftStart + (t / T) * leftGoal for t in range(T)]
rightTraj = [(1 - t / T) * rightStart + (t / T) * rightGoal for t in range(T)]

# Define contact model
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel.addContact("left_contact", crocoddyl.ContactModel6D(state, leftFootId, pinocchio.SE3.Identity(), pinocchio.LOCAL, actuation.nu, np.array([0, 0])))
contactModel.addContact("right_contact", crocoddyl.ContactModel6D(state, rightFootId, pinocchio.SE3.Identity(), pinocchio.LOCAL, actuation.nu, np.array([0, 0])))

# Joint limits (for self-collision cost)
maxfloat = sys.float_info.max
xlb = np.concatenate([-maxfloat * np.ones(7), model.lowerPositionLimit[8:], -maxfloat * np.ones(state.nv)])
xub = np.concatenate([ maxfloat * np.ones(7), model.upperPositionLimit[8:],  maxfloat * np.ones(state.nv)])
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

# State and control regularization
# xActivation = crocoddyl.ActivationModelWeightedQuad((np.array([0]*3 + [10]*3 + [0.01]*(state.nv-6) + [10]*state.nv) ** 2))
# xTActivation = crocoddyl.ActivationModelWeightedQuad((np.array([0]*3 + [10]*3 + [0.01]*(state.nv-6) + [100]*state.nv) ** 2))
xActivation  = crocoddyl.ActivationModelWeightedQuad((np.array([0]*3 + [10]*3 + [100.0]*(12) + [0.01]*(state.nv-18) + [10]*state.nv) ** 2))
xTActivation = crocoddyl.ActivationModelWeightedQuad((np.array([0]*3 + [10]*3 + [100.0]*(12) + [0.01]*(state.nv-18) + [10]*state.nv) ** 2))
xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# CoM cost
comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
comTrack = crocoddyl.CostModelResidual(state, comResidual)

# Create per-step running models with time-varying targets
runningModels = []
for t in range(T):
    # Left hand target
    target_L = leftTraj[t]
    residual_L = crocoddyl.ResidualModelFramePlacement(state, leftHandId, pinocchio.SE3(np.eye(3), target_L), actuation.nu)
    cost_L = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(np.array([1]*3 + [1e-4]*3) ** 2), residual_L)

    # Right hand target
    target_R = rightTraj[t]
    residual_R = crocoddyl.ResidualModelFramePlacement(state, rightHandId, pinocchio.SE3(np.eye(3), target_R), actuation.nu)
    cost_R = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(np.array([1]*3 + [1e-4]*3) ** 2), residual_R)

    # Combine all running costs
    costModel = crocoddyl.CostModelSum(state, actuation.nu)
    costModel.addCost("track_L", cost_L, 1e2)
    costModel.addCost("track_R", cost_R, 1e2)
    costModel.addCost("xReg", xRegCost, 1e-3)
    costModel.addCost("uReg", uRegCost, 1e-4)
    costModel.addCost("limit", limitCost, 1e3)
    costModel.addCost("comTrack", comTrack, 0.0)

    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, costModel)
    runningModel = crocoddyl.IntegratedActionModelEuler(dmodel, TIMESTEP)
    runningModels.append(runningModel)

# Terminal model using final targets for both hands
residual_L = crocoddyl.ResidualModelFramePlacement(state, leftHandId, pinocchio.SE3(np.eye(3), leftGoal), actuation.nu)
cost_L = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(np.array([1]*3 + [1e-4]*3) ** 2), residual_L)
residual_R = crocoddyl.ResidualModelFramePlacement(state, rightHandId, pinocchio.SE3(np.eye(3), rightGoal), actuation.nu)
cost_R = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(np.array([1]*3 + [1e-4]*3) ** 2), residual_R)

terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel.addCost("track_L", cost_L, 1e2)
terminalCostModel.addCost("track_R", cost_R, 1e2)
terminalCostModel.addCost("xReg", xRegTermCost, 1e-3)
terminalCostModel.addCost("limit", limitCost, 1e3)
terminalCostModel.addCost("comTrack", comTrack, 0.0)
dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0.0)

# Shooting problem
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

# Solve
solver = crocoddyl.SolverBoxFDDP(problem)
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

# Get final EE positions
xT = solver.xs[-1]
pinocchio.forwardKinematics(model, data, xT[: state.nq])
pinocchio.updateFramePlacements(model, data)
finalLeft = data.oMf[leftHandId].translation
finalRight = data.oMf[rightHandId].translation

print("Left hand reached = ({:.3f}, {:.3f}, {:.3f})".format(*finalLeft))
print("Right hand reached = ({:.3f}, {:.3f}, {:.3f})".format(*finalRight))

# Visualizing the solution in gepetto-viewer
display = None
if WITHDISPLAY:
    if display is None:
        try:
            import gepetto

            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(robot)
            display.robot.viewer.gui.addSphere(
                "world/point", 0.05, [1.0, 0.0, 0.0, 1.0]
            )  # radius = .1, RGBA=1001
            display.robot.viewer.gui.applyConfiguration(
                "world/point", [*target.tolist(), 0.0, 0.0, 0.0, 1.0]
            )  # xyz+quaternion
        except Exception:
            display = crocoddyl.MeshcatDisplay(robot)
    display.rate = -1
    display.freq = 60
    while True:
        display.displayFromSolver(solver)
        time.sleep(TIMESTEP)

# Get final state and end effector position
xT = solver.xs[-1]
pinocchio.forwardKinematics(model, data, xT[: state.nq])
pinocchio.updateFramePlacements(model, data)
com = pinocchio.centerOfMass(model, data, xT[: state.nq])
finalPosEff = np.array(
    data.oMf[model.getFrameId("L_Wrist2_Link")].translation.T.flat
)

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[1]
    plotSolution(solver, bounds=False, figIndex=1, show=False)

    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=3
    )