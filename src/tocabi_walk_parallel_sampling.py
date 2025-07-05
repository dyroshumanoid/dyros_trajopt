import os, time, numpy as np, pinocchio, crocoddyl, signal, pathlib, sys, gc
from itertools import product
from multiprocessing import Pool
from utils.custom_biped import SimpleBipedGaitProblem
from pinocchio.robot_wrapper import RobotWrapper

# -------------------- Global constants --------------------
TIMESTEP = 0.005
URDF_PATH = "/home/kwan/humanoid-trajopt-playground/robots/tocabi/robots/dyros_tocabi.urdf"
MESH_DIR = "/home/kwan/humanoid-trajopt-playground/robots/tocabi/meshes"
assert os.path.isfile(URDF_PATH), "Check URDF path"

SAVE_DIR = pathlib.Path("walking_dataset")
SAVE_DIR.mkdir(exist_ok=True)

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

# -------------------- Initial state --------------------
# q0 = np.array([
#     0,0,0,  0,0,0,1,
#     0,0,-0.24, 0.6,-0.36,0,
#     0,0,-0.24, 0.6,-0.36,0,
#     0,0,0,
#     0.3,0.3,1.5,-1.27,-1,0,-1,0,
#     0,0,
#     -0.3,-0.3,-1.5,1.27,1,0,1,0
# ])
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

model.referenceConfigurations["half_sitting"] = q0.copy()  # <-- Restore reference configuration

v0 = np.zeros(model.nv)
x0 = np.concatenate([q0, v0])

# -------------------- Gait generator ----------------------------------------
rightFoot, leftFoot = "R_Foot_Link", "L_Foot_Link"
gait = SimpleBipedGaitProblem(model, rightFoot, leftFoot)

# -------------------- Worker function --------------------
def run_ddp(cfg_tuple):
    sx, sy, sh, syaw, ssp, dsp = cfg_tuple
    try:
        pb = gait.createWalkingProblem(x0, sx, sy, syaw, sh, TIMESTEP, ssp, dsp)

        ddp = crocoddyl.SolverBoxFDDP(pb)
        ddp.th_stop = 1e-12                                     # Sets the stopping threshold: when the change in cost becomes smaller than this value, the optimization stops.

        xs = [x0] * (pb.T + 1)
        us = ddp.problem.quasiStatic(xs[:-1])
        ddp.solve(xs, us, 100, False, 0.1)

        totalKnot = ssp + dsp
        xs_valid = [ddp.xs[i] for i in range(len(ddp.xs) - 1) if i != totalKnot]
        us_valid = [ddp.us[i] for i in range(len(ddp.us))     if i != totalKnot]

        qs = np.vstack([x[:model.nq] for x in xs_valid])[:, 7:40]
        vs = np.vstack([x[model.nq:] for x in xs_valid])[:, 6:39]
        us = np.vstack(us_valid)[:, :33]
        traj = np.hstack([qs, vs, us])

        subdir = SAVE_DIR / f"dx{sx:+.2f}" / f"dy{sy:+.2f}" / f"yaw{syaw:+.2f}"
        subdir.mkdir(parents=True, exist_ok=True)
        fname = f"dx{sx:+.2f}_dy{sy:+.2f}_yaw{syaw:+.2f}_h{sh:+.2f}_ssp{ssp}_dsp{dsp}.txt"
        fpath = subdir / fname
        np.savetxt(fpath, traj, fmt="%.7f", delimiter=" ", comments="")

        del ddp, pb, xs, us
        gc.collect()
        return f"[OK] {fname}"
    except Exception as e:
        return f"[FAIL] cfg={(sx, sy, sh, syaw, ssp, dsp)} → {e}"

# -------------------- Main entry --------------------
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Define parameter grid
    # STEPX_RANGE      = np.round(np.arange(-0.3, 0.31, 0.1), 2)
    # STEPY_RANGE      = np.round(np.arange(-0.3, 0.31, 0.1), 2)
    # STEPHEIGHT_RANGE = np.round(np.arange(0.1, 0.31, 0.1), 2)
    # STEPYAW_RANGE    = np.round(np.arange(-0.3, 0.31, 0.1), 2)
    # SSP_RANGE        = np.arange(20, 121, 10)
    # DSP_RANGE        = np.arange(10, 51, 10)

    STEPX_RANGE      = [0.3]
    STEPY_RANGE      = [0.0]
    STEPHEIGHT_RANGE = [0.3]
    STEPYAW_RANGE    = [0.0]
    SSP_RANGE        = [120]
    DSP_RANGE        = [60]

    param_grid = list(product(STEPX_RANGE, STEPY_RANGE, STEPHEIGHT_RANGE, STEPYAW_RANGE, SSP_RANGE, DSP_RANGE))
    print(f"[INFO] Total parameter combinations: {len(param_grid)}")

    # Launch multiprocessing
    with Pool(processes=12) as pool:  
        results = pool.map(run_ddp, param_grid)

    # Show results
    print("\n[RESULT SUMMARY]")
    for r in results:
        print(r)

