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

from dataclasses import dataclass

import numpy as np

# Match sim: DUAL_ARM_IK_ACTION uses ik_method="dls" with the default
# ik_params lambda_val=0.01 (the 0.1 override is commented out in robot.py).
DLS_LAMBDA = 0.01

# FR3 necessary joint limits and the position-dependent velocity-limit
# parameters published in the Franka Control Interface specification. A fixed
# velocity clamp is not sufficient near a position limit: the permitted speed
# toward that limit decreases to zero as the remaining stopping distance does.
FR3_Q_MIN = np.array(
    [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
    dtype=np.float64,
)
FR3_Q_MAX = np.array(
    [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
    dtype=np.float64,
)
FR3_DQ_MAX = np.array(
    [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26],
    dtype=np.float64,
)
FR3_DQ_OFFSET = np.array(
    [0.6599, 0.2517, 0.2000, 0.3533, 0.5757, 0.4878, 0.4628],
    dtype=np.float64,
)
FR3_DDQ_DEC = np.array(
    [6.0, 2.585, 3.5, 4.0, 17.0, 5.5, 17.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class JointVelocitySolution:
    """One bounded DLS result, including values useful for safety diagnostics."""

    command: np.ndarray
    task: np.ndarray
    requested: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


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


def dls_pseudoinverse(jacobian_6x7: np.ndarray, lam: float = DLS_LAMBDA) -> np.ndarray:
    """Damped right pseudoinverse ``J^T (J J^T + lambda^2 I)^-1``."""
    J = np.asarray(jacobian_6x7, dtype=np.float64)
    JT = J.T
    lam_I = (lam ** 2) * np.eye(J.shape[0])
    return JT @ np.linalg.solve(J @ JT + lam_I, np.eye(J.shape[0]))


def dls_delta_q(jacobian_6x7: np.ndarray, delta_x: np.ndarray, lam: float = DLS_LAMBDA) -> np.ndarray:
    """Damped-least-squares joint delta. delta_q = J^T (J J^T + lam^2 I)^-1 dx."""
    J = np.asarray(jacobian_6x7, dtype=np.float64)
    dx = np.asarray(delta_x, dtype=np.float64)
    JT = J.T
    lam_I = (lam ** 2) * np.eye(J.shape[0])
    return JT @ np.linalg.solve(J @ JT + lam_I, dx)


def position_based_joint_velocity_limits(
    q: np.ndarray,
    position_margin_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """FR3 directional joint-velocity bounds at ``q``.

    ``position_margin_rad`` moves each stopping boundary inward. Motion back
    toward the range center remains possible after entering that margin; only
    motion farther toward the nearby limit is reduced to zero.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"q must have shape (7,), got {q.shape}")
    if position_margin_rad < 0.0:
        raise ValueError("position_margin_rad must be non-negative")

    upper_gap = np.maximum(0.0, (FR3_Q_MAX - position_margin_rad) - q)
    lower_gap = np.maximum(0.0, q - (FR3_Q_MIN + position_margin_rad))
    upper = np.minimum(
        FR3_DQ_MAX,
        np.maximum(0.0, -FR3_DQ_OFFSET + np.sqrt(2.0 * FR3_DDQ_DEC * upper_gap)),
    )
    lower = np.maximum(
        -FR3_DQ_MAX,
        np.minimum(0.0, FR3_DQ_OFFSET - np.sqrt(2.0 * FR3_DDQ_DEC * lower_gap)),
    )
    return lower, upper


def joint_limit_aware_dls_velocity(
    q: np.ndarray,
    jacobian_6x7: np.ndarray,
    cartesian_velocity: np.ndarray,
    *,
    lam: float = DLS_LAMBDA,
    max_joint_velocity: float = 1.0,
    position_margin_rad: float = 0.0,
    velocity_scale: float = 1.0,
    nullspace_gain: float = 0.0,
) -> JointVelocitySolution:
    """Map a Cartesian velocity to a safely bounded FR3 joint velocity.

    The task velocity is the existing DLS solution. When ``nullspace_gain`` is
    positive, a projected barrier-like bias steers redundant motion toward the
    middle of each joint range without changing the Cartesian task to first
    order. The result is then bounded by both the caller's scalar cap and the
    FR3's directional, position-dependent limits.
    """
    q = np.asarray(q, dtype=np.float64)
    J = np.asarray(jacobian_6x7, dtype=np.float64)
    v = np.asarray(cartesian_velocity, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"q must have shape (7,), got {q.shape}")
    if J.shape != (6, 7):
        raise ValueError(f"jacobian_6x7 must have shape (6, 7), got {J.shape}")
    if v.shape != (6,):
        raise ValueError(f"cartesian_velocity must have shape (6,), got {v.shape}")
    if max_joint_velocity <= 0.0:
        raise ValueError("max_joint_velocity must be positive")
    if not 0.0 < velocity_scale <= 1.0:
        raise ValueError("velocity_scale must be in (0, 1]")
    if nullspace_gain < 0.0:
        raise ValueError("nullspace_gain must be non-negative")

    J_pinv = dls_pseudoinverse(J, lam)
    task = J_pinv @ v
    requested = task.copy()
    if nullspace_gain > 0.0:
        midpoint = 0.5 * (FR3_Q_MIN + FR3_Q_MAX)
        half_range = 0.5 * (FR3_Q_MAX - FR3_Q_MIN)
        normalized = (q - midpoint) / half_range
        # The denominator makes the centering request grow near a limit. Clip
        # it before projection so it cannot dominate the Cartesian controller.
        barrier = np.maximum(1.0 - normalized * normalized, 0.1)
        center_velocity = -nullspace_gain * normalized / barrier
        center_velocity = np.clip(
            center_velocity, -max_joint_velocity, max_joint_velocity
        )
        nullspace = np.eye(7) - J_pinv @ J
        requested = requested + nullspace @ center_velocity

    requested = np.clip(requested, -max_joint_velocity, max_joint_velocity)
    lower, upper = position_based_joint_velocity_limits(q, position_margin_rad)
    lower = np.maximum(-max_joint_velocity, velocity_scale * lower)
    upper = np.minimum(max_joint_velocity, velocity_scale * upper)
    command = np.clip(requested, lower, upper)
    return JointVelocitySolution(command, task, requested, lower, upper)


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
