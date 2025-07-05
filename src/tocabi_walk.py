import os, time, numpy as np, pinocchio, crocoddyl, signal, sys
from pinocchio.robot_wrapper import RobotWrapper
from crocoddyl import MeshcatDisplay
from utils.custom_biped import SimpleBipedGaitProblem, plotSolution

# -------------------- Runtime flags -----------------------------------------
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT    = "plot"    in sys.argv or "CROCODDYL_PLOT"    in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# -------------------- Paths --------------------------------------------------
TIMESTEP = 0.002
URDF_PATH = "/home/kwan/humanoid-trajopt-playground/robots/tocabi/robots/dyros_tocabi.urdf"
MESH_DIR  = "/home/kwan/humanoid-trajopt-playground/robots/tocabi/meshes"
assert os.path.isfile(URDF_PATH), "Check URDF path"

# -------------------- Robot --------------------------------------------------
robot = RobotWrapper.BuildFromURDF(URDF_PATH, 
                                  [MESH_DIR],
                                   pinocchio.JointModelFreeFlyer())

model = robot.model
model.effortLimit *= 1.0

print("TOCABI URDF FILE IS NOW LOADED!!!")
print("nq (configuration dimension):", model.nq)
print("nv (velocity dimension):", model.nv)
print("nu (actuated DoFs):", model.nv - 6) 

# -------------------- Initial state -----------------------------------------
q0 = np.array([
    0.0, 0.0, 0.0,  0,0,0,1,
    0.0, 0.0,-0.28, 0.6,-0.32,0,
    0.0, 0.0,-0.28, 0.6,-0.32,0,
    0.0, 0.0, 0.0,
    0.3, 0.174533 , 1.22173,-1.27,-1.57, 0.0,-1.0, 0.0,
    0.0, 0.0,
   -0.3,-0.174533 ,-1.22173, 1.27, 1.57, 0.0, 1.0, 0.0
])
assert q0.size == model.nq, "Check Joint Dimension"

data = model.createData()

pinocchio.forwardKinematics(model, data, q0)
pinocchio.updateFramePlacements(model, data)
z_RF = data.oMf[model.getFrameId('R_Foot_Link')].translation[2]
z_LF = data.oMf[model.getFrameId('L_Foot_Link')].translation[2]
q0[2] -= 0.5 * (z_RF + z_LF)

model.referenceConfigurations["half_sitting"] = q0.copy()   # Reference Configuration

v0 = np.zeros(model.nv)
x0 = np.concatenate([q0, v0])

# -------------------- Gait generator ----------------------------------------
rightFoot, leftFoot = "R_Foot_Link", "L_Foot_Link"
gait = SimpleBipedGaitProblem(model, rightFoot, leftFoot)

# -------------------- Parameter ranges --------------------------------------
STEPX_RANGE   = (-0.3, 0.3)
STEPY_RANGE   = (-0.3, 0.3) 
STEPHEIGHT_RANGE   = (0.1, 0.3) 
STEPYAW_RANGE = (-0.3, 0.3) 
SSP_RANGE     = (20, 120)   # 0.2 ~ 1.2 
DSP_RANGE     = (10, 50)    # 0.1 ~ 0.5

GAITPHASES = [
    # dict(walking=dict(
    #                   stepX        = np.random.uniform(*STEPX_RANGE),
    #                   stepY        = np.random.uniform(*STEPY_RANGE),
    #                   stepYaw      = np.random.uniform(*STEPYAW_RANGE),
    #                   stepHeight   = np.random.uniform(*STEPHEIGHT_RANGE), 
    #                   timeStep=TIMESTEP,
    #                   stepKnots    = np.random.randint(*SSP_RANGE), 
    #                   supportKnots = np.random.randint(*DSP_RANGE))
    #      )
        
    dict(walking=dict(
                      stepX        = 0.3,
                      stepY        = 0.0,
                      stepYaw      = 0.0,
                      stepHeight   = 0.2, 
                      timeStep=TIMESTEP,
                      stepKnots    = 300, 
                      supportKnots = 150)
         )
    for _ in range(1)
]

# ----------------------------------------------------------
# Solve BoxFDDP
solver = []                                                 # Stores the DDP solver (ddp) for each phase for later use
for phase in GAITPHASES:                                    # Loops through each item in the GAITPHASES list.
    cfg = phase["walking"]                                  # Access the parameters (step length, step height, etc.) stored under the "walking" key.
    pb = gait.createWalkingProblem(                         # Creates a Crocoddyl ShootingProblem representing the walking task.
        x0,                                                 # x0 is the initial state (position + velocity).
        cfg["stepX"],                                  
        cfg["stepY"],                                  
        cfg["stepYaw"],                                  
        cfg["stepHeight"],
        cfg["timeStep"], 
        cfg["stepKnots"], 
        cfg["supportKnots"])
    ddp = crocoddyl.SolverBoxFDDP(pb)                       # Creates a Box-constrained DDP solver (BoxFDDP) for the problem pb.
    ddp.th_stop = 1e-12                                     # Sets the stopping threshold: when the change in cost becomes smaller than this value, the optimization stops.
    callbacks = [crocoddyl.CallbackVerbose()]               # CallbackVerbose: prints log info at each iteration (cost, gradient norm, etc.).
    if WITHPLOT:
        callbacks.append(crocoddyl.CallbackLogger())        # CallbackLogger: stores internal solver data for plotting (used only if WITHPLOT is True).
    ddp.setCallbacks(callbacks)

    xs = [x0]*(pb.T+1)                                      # Creates a list of T+1 copies of the initial state x0. 
                                                            # This is the initial guess for the state trajectory.
                                                            # pb.T : the number of control inputs
                                                            # pb.T + 1 : the number of system states
                                                            
    us = ddp.problem.quasiStatic(xs[:-1])                   # Computes quasi-static control inputs (e.g., gravity-compensating torques) for each state (excluding the terminal one). 
                                                            # Serves as the initial guess for the control trajectory.    
                                                            # x_T is the terminal state, no input is required because no control input is applied anymore
                                                            
    ddp.solve(xs, us, 100, False, 0.1)                      # Solves the optimal control problem: 
                                                            # - xs, us are the initial guesses. 
                                                            # 100: max number of iterations. 
                                                            # False: assume the initial guess is not feasible. (Whether to assume that dynamics constraints are satisfied)
                                                            # 0.1: initial regularization (to improve numerical stability).
                                                            
    totalKnot = GAITPHASES[0]["walking"]["stepKnots"] + GAITPHASES[0]["walking"]["supportKnots"]
    xs_valid = [ddp.xs[i] for i in range(len(ddp.xs) - 1) if i != totalKnot]
    us_valid = [ddp.us[i] for i in range(len(ddp.us))     if i != totalKnot]

    qs = np.vstack([x[:model.nq] for x in xs_valid])[:, 7:40]
    vs = np.vstack([x[model.nq:] for x in xs_valid])[:, 6:39]
    us = np.vstack(us_valid)[:, :33]
    traj = np.hstack([qs, vs, us])
                                                            
    solver.append(ddp)                                      # Stores the solved ddp solver in the list for later use.
    
    x0 = ddp.xs[-1]                                         # Updates x0 to be the final state of the current phase. 
                                                            # This becomes the starting point for the next phase (to ensure continuity between motions).
    
# ----------------------------------------------------------
# Save trajectories to TXT
SAVE_DIR = "./walking_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)
np.savetxt(os.path.join(SAVE_DIR, f"tocabi_walking.txt"), traj, fmt="%.6f")
print(f"Saved walking trajectories to tocabi_walking.txt'")

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
# PLOT
if WITHPLOT:
    plotSolution(solver, bounds=False, figIndex=1, show=False)
    for i, ddp in enumerate(solver):
        log = ddp.getCallbacks()[1]
        crocoddyl.plotConvergence(
            log.costs, log.pregs, log.dregs, log.grads,
            log.stops, log.steps,
            figTitle=f"walking (phase {i})", figIndex=i+3,
            show=(i == len(solver)-1))
