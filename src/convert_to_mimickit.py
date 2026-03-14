"""
convert_to_mimickit.py
======================
Convert Crocoddyl solver output (pinocchio states) to a MimicKit-compatible
.pkl motion file for the DYROS TOCABI v2 humanoid robot.

MimicKit frame format (per frame):
    [root_pos (3), root_rot_expmap (3), joint_angles (33)]
    -> total frame_dim = 39

Pinocchio free-flyer q layout:
    q = [x, y, z, qx, qy, qz, qw, joint_angles...]
         0  1  2   3   4   5   6   7 ...

Usage:
    from convert_to_mimickit import convert_solvers_to_pkl
    convert_solvers_to_pkl(solver, model, "data/motions/out.pkl", timestep=TIMESTEP)
"""

import os
import pickle
import numpy as np
import pinocchio
from mimickit.anim.motion import Motion, LoopMode, load_motion


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def quat_to_expmap(qxyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = qxyzw
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw 
    qw = np.clip(qw, -1.0, 1.0)
    half_angle = np.arccos(qw)
    angle = 2.0 * half_angle
    sin_half = np.sin(half_angle)
    if sin_half < 1e-8:
        return np.zeros(3)
    axis = np.array([qx, qy, qz]) / sin_half
    return axis * angle


# ---------------------------------------------------------------------------
# TOCABI joint order -- confirmed from actual pinocchio model output.
#
# Pinocchio parsed this URDF with L leg before R leg, and placed
# Neck/Head between the L arm and R arm chains. This is the authoritative
# order and must not be changed.
#
# Total actuated joints: 6+6+3+8+2+8 = 33
#   model.nq = 7 (freeflyer) + 33 = 40
#   model.nv = 6 (freeflyer) + 33 = 39
#   frame_dim = 3 + 3 + 33 = 39
# ---------------------------------------------------------------------------

TOCABI_JOINT_DOF = [
    # left leg (6)  -- pinocchio places L before R for this URDF
    ("L_HipYaw_Joint",      1),
    ("L_HipRoll_Joint",     1),
    ("L_HipPitch_Joint",    1),
    ("L_Knee_Joint",        1),
    ("L_AnklePitch_Joint",  1),
    ("L_AnkleRoll_Joint",   1),
    # right leg (6)
    ("R_HipYaw_Joint",      1),
    ("R_HipRoll_Joint",     1),
    ("R_HipPitch_Joint",    1),
    ("R_Knee_Joint",        1),
    ("R_AnklePitch_Joint",  1),
    ("R_AnkleRoll_Joint",   1),
    # torso (3)
    ("Waist1_Joint",        1),
    ("Waist2_Joint",        1),
    ("Upperbody_Joint",     1),
    # left arm (8)
    ("L_Shoulder1_Joint",   1),
    ("L_Shoulder2_Joint",   1),
    ("L_Shoulder3_Joint",   1),
    ("L_Armlink_Joint",     1),
    ("L_Elbow_Joint",       1),
    ("L_Forearm_Joint",     1),
    ("L_Wrist1_Joint",      1),
    ("L_Wrist2_Joint",      1),
    # head (2) -- pinocchio places Neck/Head after L arm, before R arm
    ("Neck_Joint",          1),
    ("Head_Joint",          1),
    # right arm (8)
    ("R_Shoulder1_Joint",   1),
    ("R_Shoulder2_Joint",   1),
    ("R_Shoulder3_Joint",   1),
    ("R_Armlink_Joint",     1),
    ("R_Elbow_Joint",       1),
    ("R_Forearm_Joint",     1),
    ("R_Wrist1_Joint",      1),
    ("R_Wrist2_Joint",      1),
]

TOCABI_N_JOINTS  = len(TOCABI_JOINT_DOF)        # 33
TOCABI_FRAME_DIM = 3 + 3 + TOCABI_N_JOINTS      # 39


# ---------------------------------------------------------------------------
# Joint order verification
# ---------------------------------------------------------------------------

def verify_joint_order(model) -> list:
    """
    Cross-check the loaded pinocchio model's joint order against
    TOCABI_JOINT_DOF. Raises ValueError on any mismatch so data corruption
    is caught before a single frame is written.

    Returns the verified dof_list (use this for all subsequent conversions).
    """
    model_dof_list = []
    for jid in range(1, model.njoints):
        jname = model.names[jid]
        jtype = model.joints[jid].shortname()
        if "FreeFlyer" in jtype:
            continue
        model_dof_list.append((jname, model.joints[jid].nq))

    if len(model_dof_list) != len(TOCABI_JOINT_DOF):
        raise ValueError(
            f"[verify] Joint count mismatch:\n"
            f"  pinocchio model : {len(model_dof_list)} joints\n"
            f"  TOCABI_JOINT_DOF: {len(TOCABI_JOINT_DOF)} joints\n"
            f"  Model order: {[n for n, _ in model_dof_list]}"
        )

    mismatches = []
    for i, ((m_name, m_dof), (e_name, e_dof)) in enumerate(
        zip(model_dof_list, TOCABI_JOINT_DOF)
    ):
        if m_name != e_name or m_dof != e_dof:
            mismatches.append(
                f"  [{i:2d}] model={m_name}({m_dof})"
                f" vs expected={e_name}({e_dof})"
            )
    if mismatches:
        raise ValueError(
            "[verify] Joint order mismatch:\n" + "\n".join(mismatches)
            + f"\nFull model order: {[n for n, _ in model_dof_list]}"
        )

    print(f"[verify] Joint order verified -- {len(model_dof_list)} joints OK.")
    return model_dof_list


# ---------------------------------------------------------------------------
# Single-frame converter
# ---------------------------------------------------------------------------

def q_to_mimickit_frame(q: np.ndarray, dof_list: list) -> np.ndarray:
    """
    Convert one pinocchio configuration vector -> MimicKit frame.

    q  : [x, y, z, qx, qy, qz, qw, j0, j1, ... jN]   shape (nq,)
    out: [root_pos(3), root_rot_expmap(3), joint_angles(N)]
    """
    root_pos = q[0:3].copy()
    root_rot = quat_to_expmap(q[3:7])

    joint_q = q[7:]
    joint_angles = []
    idx = 0
    for (_jname, dof) in dof_list:
        joint_angles.extend(joint_q[idx: idx + dof].tolist())
        idx += dof

    if idx != len(joint_q):
        raise ValueError(
            f"[q_to_frame] Consumed {idx} values but joint_q has {len(joint_q)}. "
            "TOCABI_JOINT_DOF does not match the loaded model."
        )

    return np.concatenate([root_pos, root_rot, joint_angles])


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def convert_solvers_to_pkl(
    solver_list,
    model,
    output_path: str,
    timestep: float = 0.01,
    fps: float = None,
):
    """
    Convert a list of solved Crocoddyl solvers into a MimicKit .pkl file.

    Args:
        solver_list : list of crocoddyl.SolverXXX  (one entry per gait phase)
        model       : pinocchio.Model  (free-flyer TOCABI model from main.py)
        output_path : destination .pkl path
        timestep    : simulation dt in seconds
        fps         : override fps; defaults to 1 / timestep
    """
    if fps is None:
        fps = 1.0 / timestep

    dof_list = verify_joint_order(model)
    expected_dim = 3 + 3 + sum(d for _, d in dof_list)
    print(
        f"[convert] frame_dim = {expected_dim}  "
        f"(3 root_pos + 3 root_rot + {sum(d for _, d in dof_list)} joints)"
    )

    frames = []
    for phase_idx, solver in enumerate(solver_list):
        before = len(frames)
        start = 1 if phase_idx > 0 else 0
        for x in solver.xs[start:-1]:
            frames.append(q_to_mimickit_frame(x[: model.nq], dof_list))
        print(
            f"  Phase {phase_idx}: {len(frames) - before} knots"
            f"  (running total: {len(frames)})"
        )

    # Terminal state of the last phase
    frames.append(
        q_to_mimickit_frame(solver_list[-1].xs[-1][: model.nq], dof_list)
    )

    frames_arr = np.array(frames, dtype=np.float32)
    assert frames_arr.shape[1] == expected_dim, (
        f"Frame dim mismatch: got {frames_arr.shape[1]}, expected {expected_dim}"
    )

    print(f"[convert] Total frames : {len(frames_arr)}")
    print(f"[convert] Duration     : {len(frames_arr) / fps:.2f} s  @ {fps:.1f} fps")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    motion = Motion(
        loop_mode=LoopMode.CLAMP,
        fps=fps,
        frames=frames_arr,
    )

    motion.save(output_path)

    print(f"[convert] Saved -> {output_path}")
    return motion


# ---------------------------------------------------------------------------
# Standalone self-test  (no pinocchio model needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    print("Standalone shape-check test...")

    dummy = np.random.randn(60, TOCABI_FRAME_DIM).astype(np.float32)

    motion = Motion(
        loop_mode=LoopMode.CLAMP,
        fps=20.0,
        frames=dummy,
    )

    out = os.path.join(tempfile.gettempdir(), "test_tocabi.pkl")
    motion.save(out)

    m = load_motion(out)

    assert m.frames.shape == (60, TOCABI_FRAME_DIM)
    print(f"OK -- frames shape: {m.frames.shape}")

    print(f"\nJoint order ({TOCABI_N_JOINTS} joints, confirmed from runtime):")
    for i, (name, dof) in enumerate(TOCABI_JOINT_DOF):
        print(f"  [{i:2d}] {name}")