#!/usr/bin/env python3
"""Wrist hand-eye calibration for the EnvFrameFranka rig.

Pick the arm with ``--arm {l,r}`` (default r). Connect that arm via EnvFrameFranka
(joint_ik), home it, then trace one circle in the env xy-plane at each of a few z
levels (centered on the arm's sweep center). The orientation aims the EE +Z axis
(the wrist-cam optical axis) at AIM_POINT from every point on the circle.

While moving, the wrist stream is shown live with detected ChArUco corners
(11x11, DICT_4X4_1000) overlaid. Whenever the board is detected (throttled by
CAPTURE_INTERVAL_S), the
clean frame is saved alongside the ACHIEVED EE pose read from the SAME
observation (env frame, xyzw) -- not the commanded pose -- into OUT_DIR.

When the sweep ends, a robot-world / hand-eye solve
(cv2.calibrateRobotWorldHandEye) recovers the wrist-cam intrinsics, the
camera-in-EE mount, and the board pose in the robot base frame. Assuming the
board is placed X-forward / Z-up aligned with the env frame (board frame == env
frame), this also yields the BOARD CENTER in the robot base. Results merge into
results/<view>.json (same store as the scene-cam camera_calibration.py path).

Run with the rig venv (third_party/franka-bimanual/.venv). No flags.
"""

import glob
import json
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Literal

import cv2 as cv
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from cv2 import aruco

from calib_common import (
    CHARUCO_SQUARES, CHARUCO_SQUARE_SIZE,
    charuco_match_env, detect_charuco, invT, load_results, quat_wxyz_from_R,
    samples_dir, save_results, T_from, fit_opencv_fisheye_to_isaac_sim_polynomial,
)
from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig
from lerobot_robot_envframe_franka.diffik import (
    FR3_Q_MAX,
    FR3_Q_MIN,
    compute_pose_error,
    joint_limit_aware_dls_velocity,
)
from lerobot_robot_envframe_franka.franka_jacobian import fk_chain, zero_jacobian

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("wrist_handeye")

EE_AXIS_KEYS = ("x", "y", "z", "qx", "qy", "qz", "qw")

# Per-arm wrist camera view name (EnvFrameFrankaConfig._DEFAULT_WRIST_CAMS).
WRIST_VIEW_BY_ARM = {"r": "wrist_right_minus", "l": "wrist_left_plus"}

# Safe joint home per arm (mirror pair; from scripts/home.py _DEFAULT_POSE).
HOME_Q_BY_ARM = {
    "r": [0.4363, -0.6109, 0.0, -2.3562, 0.0, 1.8326, -1.1345],
    "l": [-0.6109, -0.6109, 0.0, -2.3562, 0.0, 1.8326, 0.7854],
}

# Arm selected at runtime (set in main() from --arm); the module globals below
# (ARM, WRIST_VIEW, OUT_DIR, AIM_POINT, CENTER_XY, HOME_Q) are rebound there.
# ChArUco board + env re-aim live in calib_common (shared with the scene-cam
# calibration so the two can't drift): detect_charuco, charuco_match_env (matched
# object points already in the env frame, origin at board center, +X fwd/+Y
# left/+Z up), CHARUCO_SQUARES, CHARUCO_SQUARE_SIZE.
ARM = "r"

# Selected wrist cam view + its capture dir; rebound for the chosen arm in main().
# Only this cam is initialized -- no scene cams; its config is taken verbatim
# from the rig default (same IP/fps/resolution).
WRIST_VIEW = WRIST_VIEW_BY_ARM[ARM]
WINDOW = "wrist cam"

# Capture output: clean frames <idx>.png + poses.json (achieved EE pose per
# saved frame) under samples/<WRIST_VIEW>/ (shared store with the scene cams).
OUT_DIR = samples_dir(WRIST_VIEW)
# Min seconds between auto-captures, so a slow sweep past the board doesn't dump
# dozens of near-identical frames. This scales with the slower circle period so
# angular capture spacing stays approximately the same as the original
# 15s-circle / 0.5s-capture trajectory.
CAPTURE_INTERVAL_S = 1.0
# Seconds to command the first pose (streaming, NOT saving) so the arm settles
# off the home->start transit before any frame is captured (avoids motion blur).
SETTLE_S = 3.0

# Smooth, capture-free transits prevent the joint-IK loop from seeing the former
# single-tick home -> first-pose jump (about 33cm plus a large reorientation).
# The shorter inter-level transit only has to cover the 5cm z change.
HOME_TO_START_S = 6.0
BETWEEN_LEVELS_S = 4.0

# Calibration-only joint-IK settings. Keep the inner loop fast so the held pose
# remains actively tracked, but ask it to converge gently and cap each joint well
# below the normal teleop limit. The robot-level dynamics factors additionally
# bound velocity/acceleration/jerk in franky's motion generator.
CALIBRATION_CART_GAIN = 2.0
CALIBRATION_MAX_JOINT_VEL = 0.3
CALIBRATION_JOINT_IK_RELATIVE_DYNAMICS = (0.15, 0.08, 0.05)
HOME_MAX_TIME_S = 30.0
CALIBRATION_JOINT_LIMIT_MARGIN_RAD = 0.10
CALIBRATION_JOINT_LIMIT_VELOCITY_SCALE = 0.8
CALIBRATION_JOINT_LIMIT_AVOIDANCE_GAIN = 0.15
PREFLIGHT_EXTRA_MARGIN_RAD = 0.02
PREFLIGHT_POSITION_TOL_M = 0.03
PREFLIGHT_ROTATION_TOL_RAD = 0.15

# Selected after homing by simulating both equivalent look-at roll branches.
# A local-Z roll changes image roll but leaves the camera optical axis unchanged.
LOOK_AT_ROLL_RAD = 0.0

# Selected arm's joint home; rebound in main().
HOME_Q = HOME_Q_BY_ARM[ARM]

# RIGHT-arm trajectory params. The LEFT arm mirrors these across the env x-axis
# (y -> -y on AIM_POINT and CENTER_XY); main() applies the mirror per --arm.
# Point the EE +Z (wrist-cam optical axis) at this env-frame point every tick.
AIM_POINT = np.array([0, 0.05, 0.0])
# Env-frame xy center (meters) the circles are traced around (shared by all z
# levels). Set to None to use the measured home EE xy instead.
CENTER_XY = (0.1, 0.2)

# Env-frame z heights (meters) to run one circle at, in order. One circle per z.
Z_LEVELS = (0.5, 0.4, 0.3, 0.2,)

# Circle in the env xy-plane around CENTER_XY.
RADIUS = 0.1      # meters
PERIOD = 30.0      # seconds per revolution
REVOLUTIONS = 1
RAMP = 4.0         # seconds to ease radius 0->full at start and full->0 at end
FPS = 10.0


def look_at_quat(
    cam_pos: np.ndarray, target: np.ndarray, roll_rad: float | None = None
) -> np.ndarray:
    """EE orientation (quat xyzw) whose +Z axis points from cam_pos at target.

    Columns of the rotation are the EE x,y,z axes in env coords: +Z aims at the
    target (camera forward); +X is horizontal (perpendicular to the env z up-hint
    and forward); +Y completes the right-handed frame.
    """
    fwd = np.asarray(target, float) - np.asarray(cam_pos, float)
    fwd = fwd / np.linalg.norm(fwd)                 # EE +Z toward target
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, fwd)
    if np.linalg.norm(right) < 1e-6:                # forward parallel to up-hint
        right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    right = right / np.linalg.norm(right)           # EE +X
    down = np.cross(fwd, right)                      # EE +Y
    R = np.column_stack([right, down, fwd])
    # Roll in the EE's local frame. This preserves column 2 (the +Z optical
    # axis) while choosing between redundant wrist configurations. The selected
    # branch is held fixed for the run, so quaternion targets remain continuous.
    roll = LOOK_AT_ROLL_RAD if roll_rad is None else float(roll_rad)
    R = R @ Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix()
    return Rotation.from_matrix(R).as_quat()         # xyzw


def _minimum_jerk(u: float) -> float:
    """Quintic 0->1 blend with zero velocity/acceleration at both ends."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _slerp_quat(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, u: float) -> np.ndarray:
    """Shortest-path unit-quaternion interpolation (xyzw)."""
    q0 = np.asarray(q0_xyzw, dtype=np.float64)
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + float(u) * (q1 - q0)
        return q / np.linalg.norm(q)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    return (
        np.sin((1.0 - float(u)) * angle) / sin_angle * q0
        + np.sin(float(u) * angle) / sin_angle * q1
    )


@dataclass(frozen=True)
class TrajectoryTarget:
    phase: str
    pos: np.ndarray
    quat_xyzw: np.ndarray
    checkpoint: bool = False


@dataclass(frozen=True)
class PreflightResult:
    roll_rad: float
    safe: bool
    min_margin_rad: float
    limiting_joint: int
    limit_clip_count: int
    max_checkpoint_pos_error_m: float
    max_checkpoint_rot_error_rad: float

    def summary(self) -> str:
        return (
            f"roll={np.degrees(self.roll_rad):.0f}deg safe={self.safe} "
            f"min_margin={self.min_margin_rad:.3f}rad (joint {self.limiting_joint}) "
            f"limit_clips={self.limit_clip_count} checkpoint_error="
            f"{self.max_checkpoint_pos_error_m * 1000.0:.1f}mm/"
            f"{np.degrees(self.max_checkpoint_rot_error_rad):.1f}deg"
        )


def _transition_pose(
    start_pos: np.ndarray,
    start_quat_xyzw: np.ndarray,
    end_pos: np.ndarray,
    end_quat_xyzw: np.ndarray,
    step: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    blend = _minimum_jerk(step / n_steps)
    return (
        start_pos + blend * (end_pos - start_pos),
        _slerp_quat(start_quat_xyzw, end_quat_xyzw, blend),
    )


def _circle_pose(
    center: np.ndarray,
    step: int,
    n_steps: int,
    roll_rad: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    duration = PERIOD * REVOLUTIONS
    t = duration * step / n_steps
    if RAMP > 0:
        scale = min(
            _minimum_jerk(t / RAMP),
            _minimum_jerk((duration - t) / RAMP),
        )
    else:
        scale = 1.0
    theta = 2.0 * np.pi * t / PERIOD
    off = np.array(
        [
            RADIUS * scale * (np.cos(theta) - 1.0),
            RADIUS * scale * np.sin(theta),
            0.0,
        ],
        dtype=np.float64,
    )
    pos = center + off
    return pos, look_at_quat(pos, AIM_POINT, roll_rad)


def _calibration_targets(
    start_pos: np.ndarray,
    start_quat_xyzw: np.ndarray,
    center_xy: np.ndarray,
    roll_rad: float,
) -> list[TrajectoryTarget]:
    """Exact 10 Hz target sequence used to preflight one roll branch."""
    targets: list[TrajectoryTarget] = []
    transition_pos = np.asarray(start_pos, dtype=np.float64)
    transition_quat = np.asarray(start_quat_xyzw, dtype=np.float64)
    for level_idx, z in enumerate(Z_LEVELS):
        center = np.array([center_xy[0], center_xy[1], z], dtype=np.float64)
        center_quat = look_at_quat(center, AIM_POINT, roll_rad)
        duration_s = HOME_TO_START_S if level_idx == 0 else BETWEEN_LEVELS_S
        n_transition = max(1, int(round(duration_s * FPS)))
        phase = "home_to_start" if level_idx == 0 else f"level_{level_idx}_transition"
        for step in range(n_transition + 1):
            pos, quat = _transition_pose(
                transition_pos,
                transition_quat,
                center,
                center_quat,
                step,
                n_transition,
            )
            targets.append(TrajectoryTarget(phase, pos, quat))

        n_hold = max(1, int(round(SETTLE_S * FPS)))
        for step in range(n_hold):
            targets.append(
                TrajectoryTarget(
                    f"level_{level_idx}_settle",
                    center.copy(),
                    center_quat.copy(),
                    checkpoint=step == n_hold - 1,
                )
            )

        n_circle = max(1, int(round(PERIOD * REVOLUTIONS * FPS)))
        for step in range(n_circle + 1):
            pos, quat = _circle_pose(center, step, n_circle, roll_rad)
            targets.append(
                TrajectoryTarget(
                    f"level_{level_idx}_circle",
                    pos,
                    quat,
                    checkpoint=step == n_circle,
                )
            )
        transition_pos, transition_quat = center, center_quat
    return targets


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _preflight_roll(
    robot: EnvFrameFranka,
    config: EnvFrameFrankaConfig,
    q_start: np.ndarray,
    T_start_base: np.ndarray,
    start_pos_env: np.ndarray,
    start_quat_env_xyzw: np.ndarray,
    center_xy: np.ndarray,
    roll_rad: float,
) -> PreflightResult:
    """Roll out the real controller equations without sending robot commands."""
    q = np.asarray(q_start, dtype=np.float64).copy()
    T_start_base = np.asarray(T_start_base, dtype=np.float64)
    # Align the repo FK model to the measured O_T_EE once at the post-home q.
    # The residual is a fixed tool-frame transform and remains valid in rollout.
    model_start = fk_chain(q)[-1]
    model_to_measured = np.linalg.inv(model_start) @ T_start_base

    inner_steps = max(1, int(round(config.ik_hz / FPS)))
    dt = 1.0 / config.ik_hz
    min_margin = float("inf")
    limiting_joint = 0
    limit_clip_count = 0
    max_checkpoint_pos_error = 0.0
    max_checkpoint_rot_error = 0.0

    targets = _calibration_targets(
        start_pos_env, start_quat_env_xyzw, center_xy, roll_rad
    )
    for target in targets:
        T_des = robot._env_to_base(ARM, target.pos, target.quat_xyzw)
        quat_des = _quat_wxyz_from_matrix(T_des[:3, :3])
        for _ in range(inner_steps):
            T_cur = fk_chain(q)[-1] @ model_to_measured
            pos_cur = T_cur[:3, 3]
            quat_cur = _quat_wxyz_from_matrix(T_cur[:3, :3])
            J = zero_jacobian(q, pos_cur)
            dx = compute_pose_error(pos_cur, quat_cur, T_des[:3, 3], quat_des)
            solution = joint_limit_aware_dls_velocity(
                q,
                J,
                config.cart_gain * dx,
                lam=config.dls_lambda,
                max_joint_velocity=config.max_joint_vel,
                position_margin_rad=config.joint_limit_safety_margin_rad,
                velocity_scale=config.joint_limit_velocity_scale,
                nullspace_gain=config.joint_limit_avoidance_gain,
            )
            if np.any(np.abs(solution.command - solution.requested) > 1.0e-6):
                limit_clip_count += 1
            q += solution.command * dt
            margins = np.minimum(q - FR3_Q_MIN, FR3_Q_MAX - q)
            i = int(np.argmin(margins))
            if margins[i] < min_margin:
                min_margin = float(margins[i])
                limiting_joint = i + 1

        if target.checkpoint:
            T_cur = fk_chain(q)[-1] @ model_to_measured
            quat_cur = _quat_wxyz_from_matrix(T_cur[:3, :3])
            checkpoint_error = compute_pose_error(
                T_cur[:3, 3], quat_cur, T_des[:3, 3], quat_des
            )
            max_checkpoint_pos_error = max(
                max_checkpoint_pos_error,
                float(np.linalg.norm(checkpoint_error[:3])),
            )
            max_checkpoint_rot_error = max(
                max_checkpoint_rot_error,
                float(np.linalg.norm(checkpoint_error[3:])),
            )

    safe = (
        min_margin
        >= config.joint_limit_safety_margin_rad + PREFLIGHT_EXTRA_MARGIN_RAD
        and limit_clip_count == 0
        and max_checkpoint_pos_error <= PREFLIGHT_POSITION_TOL_M
        and max_checkpoint_rot_error <= PREFLIGHT_ROTATION_TOL_RAD
    )
    return PreflightResult(
        roll_rad=roll_rad,
        safe=safe,
        min_margin_rad=min_margin,
        limiting_joint=limiting_joint,
        limit_clip_count=limit_clip_count,
        max_checkpoint_pos_error_m=max_checkpoint_pos_error,
        max_checkpoint_rot_error_rad=max_checkpoint_rot_error,
    )


def _select_safe_look_at_roll(
    robot: EnvFrameFranka,
    config: EnvFrameFrankaConfig,
    q_start: np.ndarray,
    T_start_base: np.ndarray,
    start_pos_env: np.ndarray,
    start_quat_env_xyzw: np.ndarray,
    center_xy: np.ndarray,
) -> PreflightResult:
    """Choose between equivalent 0/pi optical-roll branches before moving."""
    global LOOK_AT_ROLL_RAD
    results = [
        _preflight_roll(
            robot,
            config,
            q_start,
            T_start_base,
            start_pos_env,
            start_quat_env_xyzw,
            center_xy,
            roll,
        )
        for roll in (0.0, np.pi)
    ]
    for result in results:
        logger.info("Preflight: %s", result.summary())
    valid = [result for result in results if result.safe]
    if not valid:
        details = "; ".join(result.summary() for result in results)
        raise RuntimeError(
            "No safe look-at wrist-roll branch; calibration motion was not started. "
            + details
        )
    chosen = max(valid, key=lambda result: result.min_margin_rad)
    LOOK_AT_ROLL_RAD = chosen.roll_rad
    logger.info(
        "Selected %.0fdeg local optical-axis roll (joint margin %.3frad).",
        np.degrees(chosen.roll_rad),
        chosen.min_margin_rad,
    )
    return chosen


def _pose_action(pos: np.ndarray, quat_xyzw: np.ndarray) -> dict[str, float]:
    """Build one absolute env-frame pose action for the selected arm."""
    return {
        f"{ARM}_{k}": float(v)
        for k, v in zip(EE_AXIS_KEYS, (*pos, *quat_xyzw))
    }


def _achieved_pose(obs: dict) -> np.ndarray | None:
    """Achieved EE pose (env frame) from an observation: [x,y,z,qx,qy,qz,qw].

    EnvFrameFranka observations use the sim-matched vector schema
    ``{left,right}_ee_pos`` plus ``{left,right}_ee_quat`` (WXYZ), while action
    targets use the flat ``{l,r}_*`` XYZW schema. Convert the observation here
    so every saved image has the achieved pose required by the hand-eye solve.
    """
    side = {"l": "left", "r": "right"}[ARM]
    pos = obs.get(f"{side}_ee_pos")
    quat_wxyz = obs.get(f"{side}_ee_quat")
    if pos is None or quat_wxyz is None:
        return None

    pos = np.asarray(pos, dtype=np.float64)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    if pos.shape != (3,) or quat_wxyz.shape != (4,):
        return None

    x, y, z = pos
    qw, qx, qy, qz = quat_wxyz
    return np.array([x, y, z, qx, qy, qz, qw], dtype=np.float64)


class Capturer:
    """Streams the wrist feed with board-corner overlay and auto-saves clean
    frames paired with the ACHIEVED EE pose (same observation) for hand-eye."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        # Each run starts clean: drop prior captures + poses.json so stale frames
        # never mix into the downstream calibration.
        for f in glob.glob(os.path.join(out_dir, "*.png")) + glob.glob(os.path.join(out_dir, "poses.json")):
            os.remove(f)
        self.idx = 1
        self.records: list[dict] = []   # {"image": "001.png", "pose_env_xyzw": [...]}
        self._last_save = 0.0
        self.image_size: tuple[int, int] | None = None  # (w, h) of captured frames

    def step(self, robot: EnvFrameFranka, save: bool = True) -> None:
        """One observation: show the frame (corners overlaid), and if the board
        is detected, the throttle has elapsed, and `save` is set, save the clean
        frame + pose. Pass save=False to stream without capturing (e.g. while the
        arm is still settling onto the first pose).

        Frame and pose come from the SAME get_observation() call, so the saved
        pose is the achieved EE pose at the captured frame's instant."""
        obs = robot.get_observation()
        rgb = obs.get(WRIST_VIEW)
        if rgb is None:
            return
        bgr = cv.cvtColor(np.asarray(rgb), cv.COLOR_RGB2BGR)
        if self.image_size is None:
            self.image_size = (bgr.shape[1], bgr.shape[0])  # (w, h)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        corners, ids = detect_charuco(gray)
        found = corners is not None
        pose = _achieved_pose(obs)

        view = bgr.copy()
        if found:
            aruco.drawDetectedCornersCharuco(view, corners, ids)
        color = (0, 255, 0) if found else (0, 0, 255)
        n_c = 0 if not found else len(ids)
        status = f"DETECTED ({n_c} corners)" if found else "no board"
        if not save:
            status += " (settling)"
        cv.putText(view, f'{status}  saved={len(self.records)}',
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)
        cv.imshow(WINDOW, view)
        cv.waitKey(1)

        now = time.perf_counter()
        if save and found and pose is not None and now - self._last_save >= CAPTURE_INTERVAL_S:
            name = f"{self.idx:03d}.png"
            cv.imwrite(os.path.join(self.out_dir, name), bgr)  # clean frame, no overlay
            # Recover the true base-frame O_T_EE (gripper-in-base) from the achieved
            # env pose. _env_to_base is the exact inverse of the transform that
            # produced the observation, so the base_in_env config cancels and this
            # is the real O_T_EE -- the robot-base anchor the hand-eye solve needs.
            O_T_EE = robot._env_to_base(ARM, pose[:3], pose[3:])
            self.records.append({
                "image": name,
                "pose_env_xyzw": pose.tolist(),
                "O_T_EE": O_T_EE.tolist(),  # 4x4 gripper-in-base (base frame)
            })
            self.idx += 1
            self._last_save = now
            logger.info("captured %s (achieved pose %s)", name, np.round(pose, 4))

    def save_poses(self) -> None:
        """Write poses.json: list of {image, achieved EE pose (env, xyzw)}."""
        if not self.records:
            return
        path = os.path.join(self.out_dir, "poses.json")
        with open(path, "w") as f:
            json.dump({"arm": ARM, "frame": "env", "quat_order": "xyzw",
                       "charuco_squares": list(CHARUCO_SQUARES),
                       "square_size_m": CHARUCO_SQUARE_SIZE, "records": self.records}, f, indent=2)
        logger.info("wrote %d pose(s) to %s", len(self.records), path)


def _overlay_env_axes(mtx, dist, env_T_base, cam_in_ee, used_images, gripper_T_base) -> None:
    """Draw the calibrated ENV (board-center) frame axes onto each used frame.

    `env_T_base` is the base pose in env (X output == base_in_env). The axes are
    placed via the FULL hand-eye chain, NOT a per-view PnP -- the calibration's
    own belief about where the world frame sits:
        cam_T_env = inv(cam_in_ee) @ gripper_T_base @ inv(env_T_base)
    (cam<-gripper<-base<-env). If the calibration is good, the projected origin
    lands on the board center and the axes lie along the board edges in EVERY
    view. Writes <stem>_axes.png next to each frame; never raises."""
    cam_T_gripper = invT(cam_in_ee)
    base_T_env = invT(env_T_base)
    L = 0.5 * CHARUCO_SQUARES[0] * CHARUCO_SQUARE_SIZE  # axis length ~ half board edge
    for name, g_T_b in zip(used_images, gripper_T_base):
        img = cv.imread(os.path.join(OUT_DIR, name))
        if img is None:
            continue
        cam_T_env = cam_T_gripper @ g_T_b @ base_T_env
        rvec, _ = cv.Rodrigues(cam_T_env[:3, :3])
        tvec = cam_T_env[:3, 3]
        # drawFrameAxes only understands OpenCV's pinhole distortion model.
        # Project the same origin/X/Y/Z points with the calibrated fisheye model
        # and draw the axes ourselves, so this verification uses the identical
        # projection model as the intrinsic solve.
        axis_points = np.array(
            [[0.0, 0.0, 0.0], [L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
            dtype=np.float64,
        ).reshape(-1, 1, 3)
        try:
            projected, _ = cv.fisheye.projectPoints(axis_points, rvec, tvec, mtx, dist)
        except cv.error:
            continue
        origin, x_axis, y_axis, z_axis = np.rint(projected.reshape(-1, 2)).astype(int)
        cv.line(img, tuple(origin), tuple(x_axis), (0, 0, 255), 3, cv.LINE_AA)
        cv.line(img, tuple(origin), tuple(y_axis), (0, 255, 0), 3, cv.LINE_AA)
        cv.line(img, tuple(origin), tuple(z_axis), (255, 0, 0), 3, cv.LINE_AA)
        stem = os.path.splitext(name)[0]
        cv.imwrite(os.path.join(OUT_DIR, f"{stem}_axes.png"), img)


def solve_handeye(records: list[dict], image_size: tuple[int, int]) -> dict | None:
    """Robot-world / hand-eye solve from the captured (image, O_T_EE) records.

    Eye-in-hand: the wrist cam moves with the gripper; the board is the fixed
    "world". We recover, with cv2.calibrateRobotWorldHandEye (AX = ZB):
      X = base_T_board   (robot world->board ... i.e. board pose in the base)
      Z = cam_T_gripper  (=> cam_in_ee = inv)
    using
      A = cam_T_board   from fisheye calibration on ChArUco corners (board->cam)
      B = gripper_T_base = inv(O_T_EE)             (O_T_EE is gripper-in-base)

    The board is laid X-forward / Z-up aligned with the ENV frame, and we want
    the ENV ORIGIN AT THE BOARD CENTER -- so the ChArUco object points (whose
    native origin is a board corner) are shifted by -(center) before solving.
    Then the board frame == env frame with origin at the center, base_T_board IS
    base_T_env, and the arm base-in-env transform is inv(base_T_board) (emitted
    in EnvFrameFrankaConfig's (xyz, quat_wxyz) form for direct paste-in).

    Returns a results dict (intrinsics + handeye + base_in_env), or None if too
    few usable detections.
    """
    # Per-view: detect ChArUco and match corners to object points already in the
    # ENV frame (origin at board center, +X fwd/+Y left/+Z up; via the shared
    # calib_common.charuco_match_env), then pair with the gripper pose. Because
    # the object points carry the env axes, calibrateCamera's rvecs/tvecs are
    # cam_T_env directly -- no separate center shift or BOARD_REAIM post-multiply.
    img_pts, obj_pts, gripper_T_base, used_images = [], [], [], []
    for rec in records:
        img = cv.imread(os.path.join(OUT_DIR, rec["image"]))
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners, ids = detect_charuco(gray)
        if corners is None:
            continue
        obj, ip = charuco_match_env(corners, ids)
        if obj is None:
            continue
        obj_pts.append(obj)
        img_pts.append(ip)
        gripper_T_base.append(invT(np.asarray(rec["O_T_EE"], dtype=np.float64)))
        used_images.append(rec["image"])
    n = len(obj_pts)
    if n < 3:
        logger.warning("hand-eye solve needs >=3 board detections; got %d -- skipping", n)
        return None

    # Intrinsics + per-view cam_T_env pose (A). Object points are in the env
    # frame, so fisheye.calibrate's rvecs/tvecs ARE env->cam (cam_T_env) directly.
    # The wide-angle wrist lenses are fit with OpenCV's theta-polynomial
    # fisheye model. Extending a pinhole Brown--Conrady polynomial into the
    # image periphery can make it non-invertible and fold render regions.
    # fisheye.calibrate requires one Nx1x3/Nx1x2 float64 array per view.
    fisheye_obj_pts = [points.reshape(-1, 1, 3).astype(np.float64) for points in obj_pts]
    fisheye_img_pts = [points.reshape(-1, 1, 2).astype(np.float64) for points in img_pts]
    fisheye_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv.fisheye.CALIB_FIX_SKEW
    fisheye_criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        100,
        1e-8,
    )
    rms, mtx, dist, rvecs, tvecs = cv.fisheye.calibrate(
        fisheye_obj_pts,
        fisheye_img_pts,
        image_size,
        np.eye(3, dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
        flags=fisheye_flags,
        criteria=fisheye_criteria,
    )
    logger.info("wrist intrinsics (charuco fisheye): %d view(s), reproj RMS %.3f px", n, rms)
    isaac_sim_fisheye = fit_opencv_fisheye_to_isaac_sim_polynomial(
        mtx,
        dist,
        image_size,
        image_points=np.concatenate(img_pts, axis=0),
    )
    logger.info(
        "Isaac Sim full-sensor f-theta fit: FOV %.1f deg, source RMS %.3f px, max %.3f px (%s).",
        isaac_sim_fisheye["fisheye_max_fov"],
        isaac_sim_fisheye["fit_rms_px"],
        isaac_sim_fisheye["fit_max_px"],
        isaac_sim_fisheye["fit_domain_source"],
    )

    R_board2cam = [cv.Rodrigues(r)[0] for r in rvecs]
    t_board2cam = [t.ravel() for t in tvecs]

    R_g2b = [T[:3, :3] for T in gripper_T_base]
    t_g2b = [T[:3, 3] for T in gripper_T_base]

    R_b2board, t_b2board, R_g2c, t_g2c = cv.calibrateRobotWorldHandEye(
        R_board2cam, t_board2cam, R_g2b, t_g2b,
        method=cv.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    # VERIFIED (synthetic ground truth + this dataset): for inputs A=cam_T_board,
    # B=gripper_T_base, calibrateRobotWorldHandEye's X output (R_b2board,t_b2board)
    # is the BASE POSE IN ENV directly == base_in_env (NOT base_T_env). Object
    # points are center-anchored so env origin == board center.
    env_T_base = T_from(t_b2board, R_b2board)        # base pose in env == base_in_env (X)
    base_T_env = invT(env_T_base)                    # env(board-center) pose in base
    cam_T_gripper = T_from(t_g2c, R_g2c)
    cam_in_ee = invT(cam_T_gripper)                  # wrist cam mount on the EE

    # Board CENTER in the base frame is now just base_T_env's translation (env
    # origin == board center).
    center_base = base_T_env[:3, 3]
    logger.info("board CENTER in robot base (m): %s", np.round(center_base, 4))

    bie_t = env_T_base[:3, 3]
    bie_qw, bie_qx, bie_qy, bie_qz = quat_wxyz_from_R(env_T_base[:3, :3])
    logger.info("base_in_env[%s]: ((%.5f, %.5f, %.5f), (%.5f, %.5f, %.5f, %.5f))",
                ARM, *bie_t, bie_qw, bie_qx, bie_qy, bie_qz)

    # Verification overlay: draw the calibrated env frame on each used frame via
    # the full hand-eye chain (cam<-gripper<-base<-env). Lands on the board
    # center if the calibration is right. Takes X (base-pose-in-env == env_T_base);
    # it inverts internally to build cam_T_env.
    _overlay_env_axes(mtx, dist, env_T_base, cam_in_ee, used_images, gripper_T_base)

    qw, qx, qy, qz = quat_wxyz_from_R(base_T_env[:3, :3])
    cqw, cqx, cqy, cqz = quat_wxyz_from_R(cam_in_ee[:3, :3])
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "intrinsics": {
            "source": "charuco_fisheye_calibrate",
            "model": "opencv_fisheye",
            "matrix": mtx.tolist(),
            "distortion": dist.ravel().tolist(),
            "distortion_order": ["k1", "k2", "k3", "k4"],
            "reproj_rms_px": float(rms),
            "n_views": n,
        },
        # Ready to paste into Isaac Sim's legacy FisheyeCameraCfg. The source
        # OpenCV fisheye coefficients use a different radial parameterization.
        "isaac_sim_fisheye_polynomial": isaac_sim_fisheye,
        "handeye": {
            "method": "shah",
            "n_views": n,
            # env(=board-center) pose in the robot base frame.
            "base_T_env": base_T_env.tolist(),
            "base_to_env_translation_xyz": base_T_env[:3, 3].tolist(),
            "base_to_env_quaternion_wxyz": [qw, qx, qy, qz],
            # wrist camera mount on the EE (gripper) frame.
            "cam_in_ee_matrix": cam_in_ee.tolist(),
            "cam_in_ee_translation_xyz": cam_in_ee[:3, 3].tolist(),
            "cam_in_ee_quaternion_wxyz": [cqw, cqx, cqy, cqz],
        },
        # The recovered arm transform, ready to paste into
        # EnvFrameFrankaConfig.base_in_env[arm] = (xyz, quat_wxyz).
        "base_in_env": {
            "arm": ARM,
            "translation_xyz": bie_t.tolist(),
            "quaternion_wxyz": [bie_qw, bie_qx, bie_qy, bie_qz],
        },
        # Board center (== env origin) in the base frame.
        "board_center_in_base_xyz": center_base.tolist(),
    }


def stream_pose_transition(
    robot: EnvFrameFranka,
    capturer: "Capturer",
    start_pos: np.ndarray,
    start_quat_xyzw: np.ndarray,
    end_pos: np.ndarray,
    end_quat_xyzw: np.ndarray,
    duration_s: float,
    label: str,
) -> None:
    """Stream a capture-free minimum-jerk Cartesian pose transition.

    Position follows a quintic blend and orientation follows shortest-path SLERP
    evaluated at the same blend value. Consequently the pose target starts and
    ends with zero velocity and acceleration instead of jumping in one IK tick.
    """
    start_pos = np.asarray(start_pos, dtype=np.float64)
    end_pos = np.asarray(end_pos, dtype=np.float64)
    start_quat_xyzw = np.asarray(start_quat_xyzw, dtype=np.float64)
    end_quat_xyzw = np.asarray(end_quat_xyzw, dtype=np.float64)
    n_steps = max(1, int(round(duration_s * FPS)))
    period = 1.0 / FPS
    logger.info("%s over %.1fs (%d target steps, no capture).", label, duration_s, n_steps)

    for step in range(n_steps + 1):
        t0 = time.perf_counter()
        cmd_pos, cmd_quat = _transition_pose(
            start_pos,
            start_quat_xyzw,
            end_pos,
            end_quat_xyzw,
            step,
            n_steps,
        )
        robot.send_action(_pose_action(cmd_pos, cmd_quat))
        capturer.step(robot, save=False)
        robot.check_motion_health()

        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)


def hold_pose(
    robot: EnvFrameFranka,
    capturer: "Capturer",
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    duration_s: float,
) -> None:
    """Hold a pose for settling while streaming video but taking no samples."""
    action = _pose_action(pos, quat_xyzw)
    deadline = time.perf_counter() + duration_s
    period = 1.0 / FPS
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        robot.send_action(action)
        capturer.step(robot, save=False)
        robot.check_motion_health()
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)


def trace_circle(robot: EnvFrameFranka, center: np.ndarray, capturer: "Capturer") -> None:
    """Trace one ramped xy-plane circle around `center`, EE +Z aimed at AIM_POINT.

    `center` is the env-frame circle center (xy from home, z = this level). The
    ramp eases the radius 0->full->0, so the path starts and ends exactly at
    `center` (no velocity jolt entering/leaving a level).
    """
    duration = PERIOD * REVOLUTIONS
    n_steps = max(1, int(round(duration * FPS)))
    period = 1.0 / FPS

    # Include the final endpoint so the next level transition starts from exactly
    # the same center pose instead of inheriting a residual circle offset.
    for step in range(n_steps + 1):
        t0 = time.perf_counter()
        cmd_pos, quat_xyzw = _circle_pose(center, step, n_steps)
        robot.send_action(_pose_action(cmd_pos, quat_xyzw))
        capturer.step(robot)
        robot.check_motion_health()

        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)


@dataclass
class Args:
    arm: Literal["l", "r"] 
    """which arm to calibrate: r = RIGHT (luigi), l = LEFT (mario). The left
    trajectory is the right one mirrored across the env x-axis (y -> -y)."""


def _select_arm(arm: str) -> None:
    """Rebind the module globals the rest of the script reads to `arm`. The left
    arm mirrors the right trajectory across the env x-axis (negate y)."""
    global ARM, WRIST_VIEW, OUT_DIR, HOME_Q, AIM_POINT, CENTER_XY
    ARM = arm
    WRIST_VIEW = WRIST_VIEW_BY_ARM[arm]
    OUT_DIR = samples_dir(WRIST_VIEW)
    HOME_Q = HOME_Q_BY_ARM[arm]
    if arm == "l":
        # Mirror the right-arm sweep across the env x-axis: y -> -y.
        AIM_POINT = AIM_POINT * np.array([1.0, -1.0, 1.0])
        if CENTER_XY is not None:
            CENTER_XY = (CENTER_XY[0], -CENTER_XY[1])


def main() -> None:
    args = tyro.cli(Args)
    _select_arm(args.arm)
    logger.info("Calibrating %s arm -> view %s", ARM, WRIST_VIEW)

    # Initialize ONLY the selected wrist cam: build the default rig, then keep
    # just its wrist-cam config (verbatim IP/fps/resolution) and drop scene cams.
    default_cams = EnvFrameFrankaConfig(active_arms=(ARM,), enable_cameras=True).cameras
    cfg = EnvFrameFrankaConfig(
        active_arms=(ARM,),
        control_mode="joint_ik",
        cart_gain=CALIBRATION_CART_GAIN,
        max_joint_vel=CALIBRATION_MAX_JOINT_VEL,
        joint_limit_safety_margin_rad=CALIBRATION_JOINT_LIMIT_MARGIN_RAD,
        joint_limit_velocity_scale=CALIBRATION_JOINT_LIMIT_VELOCITY_SCALE,
        joint_limit_avoidance_gain=CALIBRATION_JOINT_LIMIT_AVOIDANCE_GAIN,
        abort_on_joint_limit=True,
        raise_on_motion_error=True,
        joint_ik_relative_dynamics=CALIBRATION_JOINT_IK_RELATIVE_DYNAMICS,
        enable_cameras=True,
        cameras={WRIST_VIEW: default_cams[WRIST_VIEW]},
    )
    robot = EnvFrameFranka(cfg)
    capturer = Capturer(OUT_DIR)

    try:
        robot.connect()
        logger.info("Homing %s arm.", ARM)
        homed = robot.home(
            {ARM: np.asarray(HOME_Q, dtype=np.float64)},
            max_time_s=HOME_MAX_TIME_S,
        )
        if not homed:
            raise RuntimeError(
                f"{ARM} arm did not reach home within {HOME_MAX_TIME_S:.1f}s; "
                "calibration motion was not started."
            )

        # xy circle center: CENTER_XY if set, else the measured home EE xy.
        q_start, T_start_base = robot.robot_manager.current_ik_state_batch([ARM])[ARM]
        start = robot._base_to_env(ARM, T_start_base)
        center_xy = (np.asarray(CENTER_XY, dtype=np.float64) if CENTER_XY is not None
                     else np.asarray(start[0], dtype=np.float64)[:2].copy())
        logger.info("Home EE pos (env): %s; center_xy=%s; aiming EE +Z at %s",
                    start[0], center_xy, AIM_POINT)
        logger.info("Preflighting equivalent look-at wrist-roll branches.")
        _select_safe_look_at_roll(
            robot,
            cfg,
            q_start,
            T_start_base,
            np.asarray(start[0], dtype=np.float64),
            np.asarray(start[1], dtype=np.float64),
            center_xy,
        )
        logger.info(
            "Tracing %d circle(s) at z=%s, %.1f rev(s) each at %.1f Hz, r=%.3fm "
            "(Ctrl-C to stop).",
            len(Z_LEVELS), Z_LEVELS, REVOLUTIONS, FPS, RADIUS,
        )

        # Smoothly reach each level with capture disabled. The first transit starts
        # from the measured home pose; later ones start from the exact center pose
        # commanded at the end of the preceding circle, preserving target continuity.
        transition_pos = np.asarray(start[0], dtype=np.float64)
        transition_quat = np.asarray(start[1], dtype=np.float64)
        for level_idx, z in enumerate(Z_LEVELS):
            center = np.array([center_xy[0], center_xy[1], z], dtype=np.float64)
            center_quat = look_at_quat(center, AIM_POINT)
            duration_s = HOME_TO_START_S if level_idx == 0 else BETWEEN_LEVELS_S
            label = "Moving smoothly from home to first calibration pose" if level_idx == 0 else (
                f"Moving smoothly to calibration level z={z:.3f}"
            )
            stream_pose_transition(
                robot,
                capturer,
                transition_pos,
                transition_quat,
                center,
                center_quat,
                duration_s,
                label,
            )
            logger.info("Settling at z=%.3f for %.1fs before capture.", z, SETTLE_S)
            hold_pose(robot, capturer, center, center_quat, SETTLE_S)
            logger.info("Circle at z=%.3f (center=%s).", z, center)
            trace_circle(robot, center, capturer)
            transition_pos, transition_quat = center, center_quat

        logger.info(
            "Collection trajectory complete: traced %d circle(s); captured %d frame(s); "
            "image_size=%s.",
            len(Z_LEVELS),
            len(capturer.records),
            capturer.image_size,
        )
    except KeyboardInterrupt:
        logger.info("Ctrl-C received; stopping the arm and disconnecting.")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            capturer.save_poses()
        except Exception:
            logger.exception("error saving poses")
        cv.destroyAllWindows()
        try:
            robot.disconnect()
        except Exception:
            logger.exception("error disconnecting robot")

    # Arm is down; run the hand-eye solve and persist results/<view>.json (same
    # results store as the scene-cam camera_calibration.py path).
    if capturer.image_size is None:
        logger.warning("Hand-eye solve skipped: no wrist-camera frames were received.")
    elif not capturer.records:
        logger.warning(
            "Hand-eye solve skipped: collected zero usable ChArUco detections. "
            "Check the live overlay and board visibility."
        )
    else:
        logger.info(
            "Starting hand-eye solve from %d captured frame(s) at image_size=%s.",
            len(capturer.records),
            capturer.image_size,
        )
        try:
            result = solve_handeye(capturer.records, capturer.image_size)
            if result is not None:
                data = load_results(WRIST_VIEW)
                data.update(result)
                save_results(WRIST_VIEW, data)
            else:
                logger.warning("Hand-eye solve returned no result.")
        except Exception:
            logger.exception("hand-eye solve failed")
            print("hand-eye solve failed; traceback follows:", file=sys.stdout, flush=True)
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
