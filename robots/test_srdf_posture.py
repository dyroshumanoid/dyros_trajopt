#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper
import crocoddyl
from crocoddyl import MeshcatDisplay


# ============ User-configurable paths ======================================
# You can pass URDF path as the first argument, otherwise edit the default below.
if len(sys.argv) > 1:
    URDF_PATH = sys.argv[1]
else:
    # TODO: Set this to your actual dyros_tocabi_v2 URDF path
    # Example:
    # URDF_PATH = "/home/kwan/dyros_trajopt/robots/dyros_tocabi_v2/dyros_tocabi_v2.urdf"
    # URDF_PATH = "./tocabi/urdf/dyros_tocabi.urdf"
    URDF_PATH = "./p73/urdf/p73.urdf"
    # URDF_PATH = "./simple_humanoid/urdf/simple_humanoid.urdf"


# SRDF is assumed to be in the same folder with the same base name
SRDF_PATH = URDF_PATH.replace(".urdf", ".srdf")

# Mesh directory: you can either point to the same folder as URDF,
# or to the top-level mesh directory if you have one.
MESH_DIR = os.path.dirname(URDF_PATH)


# ============ Build robot model ============================================
if not os.path.isfile(URDF_PATH):
    raise FileNotFoundError(f"[test_srdf] URDF not found: {URDF_PATH}")

if not os.path.isfile(SRDF_PATH):
    raise FileNotFoundError(f"[test_srdf] SRDF not found: {SRDF_PATH}")

print("[test_srdf] URDF :", URDF_PATH)
print("[test_srdf] SRDF :", SRDF_PATH)

# Build robot with a free-flyer joint
robot = RobotWrapper.BuildFromURDF(
    URDF_PATH,
    [MESH_DIR],
    pinocchio.JointModelFreeFlyer()
)
model = robot.model
data = model.createData()

print("========== MODEL INFO ==========")
print("nq:", model.nq)
print("nv:", model.nv)

# ============ Load SRDF reference configurations ============================
pinocchio.loadReferenceConfigurations(model, SRDF_PATH)
print("========== REFERENCE CONFIGURATIONS ==========")
print("========== REFERENCE CONFIGURATIONS ==========")
ref_names = [name for name in model.referenceConfigurations]
print(ref_names)

# Postures we want to test
posture_names = [
    # "half_sitting",
    # "flying_ready",
    # "flying_takeoff",
    # "flying",
    # "flying_land",
    "side_sitting",
    "side_takeoff",
    "side_flying",
    "side_inverse",
    "side_landing"
]

# Filter only those that actually exist in the SRDF
posture_names = [n for n in posture_names if n in model.referenceConfigurations]
if not posture_names:
    raise RuntimeError("[test_srdf] No matching postures found in referenceConfigurations.")

print("Will display postures:", posture_names)


# ============ Meshcat display ==============================================
display = MeshcatDisplay(robot)
nq = model.nq

def show_posture(name: str, duration: float = 2.0):
    """Display a single reference configuration in Meshcat for a fixed duration."""
    print(f"[test_srdf] Showing posture: {name}")
    q = model.referenceConfigurations[name].copy()
    if q.size != nq:
        raise RuntimeError(
            f"[test_srdf] Posture '{name}' has size {q.size} but model.nq is {nq}"
        )

    # Run a short loop just to keep Meshcat updated for 'duration' seconds
    t0 = time.time()
    while time.time() - t0 < duration:
        display.robot.display(q)
        time.sleep(0.03)


# ============ Main loop ====================================================
if __name__ == "__main__":
    print("\n========== START MESHcat VISUALIZATION ==========")
    print("A browser window should open automatically (if not already open).")
    print("Press Ctrl+C in this terminal to exit.\n")

    try:
        while True:
            for name in posture_names:
                show_posture(name, duration=2.0)
            # After cycling all postures, pause a bit
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[test_srdf] Stopped by user.")
