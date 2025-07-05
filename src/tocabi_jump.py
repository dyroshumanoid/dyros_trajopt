import os, time, numpy as np, pinocchio, crocoddyl, signal, sys
from pinocchio.robot_wrapper import RobotWrapper
from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution
# from pinocchio.visualize import MeshcatVisualizer
from crocoddyl import MeshcatDisplay

# ----------------------------------------------------------
# 1) 플래그
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT    = "plot"    in sys.argv or "CROCODDYL_PLOT"    in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# ----------------------------------------------------------
# 2) URDF 로드
TIMESTEP = 0.01
urdf_path = "/home/kwan/humanoid-trajopt-playground/robots/tocabi/robots/dyros_tocabi.urdf"
mesh_dir  = "/home/kwan/humanoid-trajopt-playground/robots/tocabi/meshes"
assert os.path.isfile(urdf_path), "URDF 경로 확인"

robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir],
                                   pinocchio.JointModelFreeFlyer())
model = robot.model
model.effortLimit *= 1.0

# ----------------------------------------------------------
# 3) 초기 상태(q0, v0)
q0 = np.array([
    0,0,0,  0,0,0,1,
    0,0,-0.24, 0.6,-0.36,0,
    0,0,-0.24, 0.6,-0.36,0,
    0,0,0,
    0.3,0.3,1.5,-1.27,-1,0,-1,0,
    0,0,
    -0.3,-0.3,-1.5,1.27,1,0,1,0
])
assert q0.size == model.nq

data = model.createData()
pinocchio.forwardKinematics(model, data, q0)
pinocchio.updateFramePlacements(model, data)
z_RF = data.oMf[model.getFrameId('R_Foot_Link')].translation[2]
z_LF = data.oMf[model.getFrameId('L_Foot_Link')].translation[2]
z0   = 0.5 * (z_RF + z_LF)
q0[2] -= z0

model.referenceConfigurations["half_sitting"] = q0.copy()

v0 = np.zeros(model.nv)
x0 = np.concatenate([q0, v0])

# ----------------------------------------------------------
# 4) 걷기 문제 생성
rightFoot, leftFoot = "R_Foot_Link", "L_Foot_Link"
gait = SimpleBipedGaitProblem(model, rightFoot, leftFoot, fwddyn=True)

GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.03,
            "stepKnots": 15,
            "supportKnots": 4,
        }
    },
    {
        "jumping": {
            "jumpHeight": 0.1,
            "jumpLength": [0.0, 0.3, 0.0],
            "timeStep": 0.03,
            "groundKnots": 9,
            "flyingKnots": 6,
        }
    },
    {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.03,
            "stepKnots": 15,
            "supportKnots": 2,
        }
    },
]

solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            # Creating a walking problem
            solver[i] = crocoddyl.SolverIntro(
                gait.createWalkingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "jumping":
            # Creating a jumping problem
            solver[i] = crocoddyl.SolverIntro(
                gait.createJumpingProblem(
                    x0,
                    value["jumpHeight"],
                    value["jumpLength"],
                    value["timeStep"],
                    value["groundKnots"],
                    value["flyingKnots"],
                )
            )
        solver[i].th_stop = 1e-7

    # Added the callback functions
    print("*** SOLVE " + key + " ***")
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
# 5) Meshcat 시각화
if WITHDISPLAY:
    display = MeshcatDisplay(robot)   # robot = RobotWrapper
    display.forceScale = 1.0      # 접촉 힘 벡터 크기 스케일
    display.frictionConeScale = 0.07  # 마찰 원뿔 크기
    display.withContactForces = True
    display.withFrictionCones = True
    display.rate = -1                 # 모든 스텝 그리기
    display.freq = 1
    while True:
        for ddp in solver:
            display.displayFromSolver(ddp)
        time.sleep(1.0)

# ----------------------------------------------------------
# 6) 로그 및 플롯
if WITHPLOT:
    plotSolution(solver, bounds=False, figIndex=1, show=False)
    for i, ddp in enumerate(solver):
        log = ddp.getCallbacks()[1]
        crocoddyl.plotConvergence(
            log.costs, log.pregs, log.dregs, log.grads,
            log.stops, log.steps,
            figTitle=f"walking (phase {i})", figIndex=i+3,
            show=(i == len(solver)-1))
