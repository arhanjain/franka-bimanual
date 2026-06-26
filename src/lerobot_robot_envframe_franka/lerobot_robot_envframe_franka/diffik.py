"""Damped-least-squares differential IK, ported 1:1 from the sim controller.

Faithful numpy port of isaaclab ``DifferentialIKController`` (dls method) +
``compute_pose_error`` (axis_angle) so the REAL arm runs the SAME law as the sim
``LBM-Scenario-ImplicitIK-State`` stack:

    delta_q = J^T (J J^T + lambda^2 I)^-1 * delta_x        (lambda = 0.01)
    q_des   = q + delta_q

where ``delta_x`` is the 6-vector [position_error, axis_angle_error]. The result
is a JOINT POSITION step ``delta_q``; the envframe joint_ik loop streams it to
franky as a joint *velocity* (``delta_q * ik_hz``) -- a joint-position target via
``JointMotion`` can't be streamed through franky without the arm sagging (see
franka_link.py), but the redundancy resolution here is identical to sim either
way, so the joint-space trajectory still matches.

Pure functions, no hardware imports: unit-testable on the workstation and
bit-comparable against the sim controller on identical (J, delta_x) inputs.

Quaternion convention is WXYZ throughout (sim/isaaclab convention).

Vendored from ``nuc_server/nuc_server/diffik.py`` (verified bit-faithful to the
isaaclab controller by ``nuc_server.test_diffik``) so the workstation envframe
stack stays self-contained. Keep the two copies in sync if either changes.
"""

from __future__ import annotations

import numpy as np

# Match sim: DUAL_ARM_IK_ACTION uses ik_method="dls" with the default
# ik_params lambda_val=0.01 (the 0.1 override is commented out in robot.py).
DLS_LAMBDA = 0.01


def axis_angle_from_quat(quat_wxyz: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Quaternion (w, x, y, z) -> axis-angle 3-vector. Port of isaaclab.

    Magnitude is the angle (rad) turned anti-clockwise about the axis. Uses the
    same hemisphere flip (w<0) and small-angle Taylor branch as the sim.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    # Flip to the w >= 0 hemisphere (shortest rotation), as in sim.
    q = q * (1.0 - 2.0 * (q[..., 0:1] < 0.0))
    mag = np.linalg.norm(q[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, q[..., 0])
    angle = 2.0 * half_angle
    sin_half_over_angle = np.where(
        np.abs(angle) > eps,
        np.sin(half_angle) / np.where(angle == 0.0, 1.0, angle),
        0.5 - angle * angle / 48.0,
    )
    return q[..., 1:4] / sin_half_over_angle[..., None]


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two WXYZ quaternions (standard formulation)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def compute_pose_error(
    pos_cur: np.ndarray, quat_cur_wxyz: np.ndarray,
    pos_des: np.ndarray, quat_des_wxyz: np.ndarray,
) -> np.ndarray:
    """6-vector pose error [pos_err(3), axis_angle_err(3)], matching sim.

    q_error = q_des * q_cur^-1, then axis-angle. Position error is des - cur.
    """
    pos_err = np.asarray(pos_des, dtype=np.float64) - np.asarray(pos_cur, dtype=np.float64)
    qc = np.asarray(quat_cur_wxyz, dtype=np.float64)
    qd = np.asarray(quat_des_wxyz, dtype=np.float64)
    # q_cur^-1 = conj(q_cur) / |q_cur|^2
    norm = _quat_mul(qc, _quat_conjugate(qc))[0]
    q_cur_inv = _quat_conjugate(qc) / norm
    quat_err = _quat_mul(qd, q_cur_inv)
    aa_err = axis_angle_from_quat(quat_err)
    return np.concatenate([pos_err, aa_err])


def dls_delta_q(jacobian_6x7: np.ndarray, delta_x: np.ndarray, lam: float = DLS_LAMBDA) -> np.ndarray:
    """Damped-least-squares joint delta. delta_q = J^T (J J^T + lam^2 I)^-1 dx."""
    J = np.asarray(jacobian_6x7, dtype=np.float64)
    dx = np.asarray(delta_x, dtype=np.float64)
    JT = J.T
    lam_I = (lam ** 2) * np.eye(J.shape[0])
    return JT @ np.linalg.solve(J @ JT + lam_I, dx)


def joint_position_target(
    q: np.ndarray,
    jacobian_6x7: np.ndarray,
    pos_cur: np.ndarray, quat_cur_wxyz: np.ndarray,
    pos_des: np.ndarray, quat_des_wxyz: np.ndarray,
    lam: float = DLS_LAMBDA,
    max_delta_q: np.ndarray | float | None = None,
) -> np.ndarray:
    """One DLS-IK step: current state + desired EE pose -> q_des = q + delta_q.

    ``max_delta_q`` (per-joint rad or scalar) clamps the step as a real-hardware
    safety net -- the sim's max_joint_vel clamp is dead code, but on real
    hardware we cap |delta_q| to avoid singularity-driven velocity spikes. Pass
    ``max_joint_vel * dt`` for a velocity limit.
    """
    dx = compute_pose_error(pos_cur, quat_cur_wxyz, pos_des, quat_des_wxyz)
    dq = dls_delta_q(jacobian_6x7, dx, lam)
    if max_delta_q is not None:
        dq = np.clip(dq, -max_delta_q, max_delta_q)
    return np.asarray(q, dtype=np.float64) + dq
