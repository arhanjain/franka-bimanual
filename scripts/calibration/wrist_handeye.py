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
import time
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
    samples_dir, save_results, T_from,
)
from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("wrist_handeye")

EE_AXIS_KEYS = ("x", "y", "z", "qx", "qy", "qz", "qw")

# Per-arm wrist camera view name (EnvFrameFrankaConfig._DEFAULT_WRIST_CAMS).
WRIST_VIEW_BY_ARM = {"r": "wrist_right_minus", "l": "wrist_left_plus"}

# Safe joint home per arm (mirror pair; from scripts/home.py _DEFAULT_POSE).
HOME_Q_BY_ARM = {
    "r": [0.6109, -0.6109, 0.0, -2.3562, 0.0, 1.8326, -0.7854],
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
# dozens of near-identical frames.
CAPTURE_INTERVAL_S = 0.5
# Seconds to command the first pose (streaming, NOT saving) so the arm settles
# off the home->start transit before any frame is captured (avoids motion blur).
SETTLE_S = 3.0

# Selected arm's joint home; rebound in main().
HOME_Q = HOME_Q_BY_ARM[ARM]

# RIGHT-arm trajectory params. The LEFT arm mirrors these across the env x-axis
# (y -> -y on AIM_POINT and CENTER_XY); main() applies the mirror per --arm.
# Point the EE +Z (wrist-cam optical axis) at this env-frame point every tick.
AIM_POINT = np.array([-0.1, -0.1, 0.0])
# Env-frame xy center (meters) the circles are traced around (shared by all z
# levels). Set to None to use the measured home EE xy instead.
CENTER_XY = (-0.1, 0.1)

# Env-frame z heights (meters) to run one circle at, in order. One circle per z.
Z_LEVELS = (0.55, 0.5)

# Circle in the env xy-plane around CENTER_XY.
RADIUS = 0.1      # meters
PERIOD = 15.0       # seconds per revolution
REVOLUTIONS = 1
RAMP = 2.0         # seconds to ease radius 0->full at start and full->0 at end
FPS = 10.0


def look_at_quat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    return Rotation.from_matrix(R).as_quat()         # xyzw


def _achieved_pose(obs: dict) -> np.ndarray | None:
    """Achieved EE pose (env frame) from an observation: [x,y,z,qx,qy,qz,qw]."""
    keys = [f"{ARM}_{k}" for k in EE_AXIS_KEYS]
    if not all(k in obs for k in keys):
        return None
    return np.array([float(obs[k]) for k in keys], dtype=np.float64)


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
        try:
            cv.drawFrameAxes(img, mtx, dist, rvec, tvec, L, 3)
        except cv.error:
            continue
        stem = os.path.splitext(name)[0]
        cv.imwrite(os.path.join(OUT_DIR, f"{stem}_axes.png"), img)


def solve_handeye(records: list[dict], image_size: tuple[int, int]) -> dict | None:
    """Robot-world / hand-eye solve from the captured (image, O_T_EE) records.

    Eye-in-hand: the wrist cam moves with the gripper; the board is the fixed
    "world". We recover, with cv2.calibrateRobotWorldHandEye (AX = ZB):
      X = base_T_board   (robot world->board ... i.e. board pose in the base)
      Z = cam_T_gripper  (=> cam_in_ee = inv)
    using
      A = cam_T_board   from calibrateCamera on ChArUco corners (board->cam)
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
    # frame, so calibrateCamera's rvecs/tvecs ARE env->cam (cam_T_env) directly.
    rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_pts, img_pts, image_size, None, None)
    logger.info("wrist intrinsics (charuco): %d view(s), reproj RMS %.3f px", n, rms)

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
            "source": "charuco_calibrateCamera",
            "matrix": mtx.tolist(),
            "distortion": dist.ravel().tolist(),
            "reproj_rms_px": float(rms),
            "n_views": n,
        },
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


def trace_circle(robot: EnvFrameFranka, center: np.ndarray, capturer: "Capturer") -> None:
    """Trace one ramped xy-plane circle around `center`, EE +Z aimed at AIM_POINT.

    `center` is the env-frame circle center (xy from home, z = this level). The
    ramp eases the radius 0->full->0, so the path starts and ends exactly at
    `center` (no velocity jolt entering/leaving a level).
    """
    omega = 2.0 * np.pi / PERIOD
    duration = PERIOD * REVOLUTIONS
    n_steps = int(round(duration * FPS))
    period = 1.0 / FPS

    for step in range(n_steps):
        t0 = time.perf_counter()
        t = step / FPS

        scale = min(1.0, t / RAMP, max(0.0, (duration - t) / RAMP)) if RAMP > 0 else 1.0
        theta = omega * t
        off = np.zeros(3, dtype=np.float64)
        off[0] = RADIUS * scale * (np.cos(theta) - 1.0)  # env x
        off[1] = RADIUS * scale * np.sin(theta)          # env y

        cmd_pos = center + off
        quat_xyzw = look_at_quat(cmd_pos, AIM_POINT)  # EE +Z aims at AIM_POINT
        action = {f"{ARM}_{k}": float(v)
                  for k, v in zip(EE_AXIS_KEYS, (*cmd_pos, *quat_xyzw))}
        robot.send_action(action)
        capturer.step(robot)

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
        enable_cameras=True,
        cameras={WRIST_VIEW: default_cams[WRIST_VIEW]},
    )
    robot = EnvFrameFranka(cfg)
    capturer = Capturer(OUT_DIR)

    try:
        robot.connect()
        logger.info("Homing %s arm.", ARM)
        robot.home({ARM: np.asarray(HOME_Q, dtype=np.float64)})

        # xy circle center: CENTER_XY if set, else the measured home EE xy.
        start = robot.current_ee_pose_env()[ARM]
        center_xy = (np.asarray(CENTER_XY, dtype=np.float64) if CENTER_XY is not None
                     else np.asarray(start[0], dtype=np.float64)[:2].copy())
        logger.info("Home EE pos (env): %s; center_xy=%s; aiming EE +Z at %s",
                    start[0], center_xy, AIM_POINT)
        logger.info(
            "Tracing %d circle(s) at z=%s, %.1f rev(s) each at %.1f Hz, r=%.3fm "
            "(Ctrl-C to stop).",
            len(Z_LEVELS), Z_LEVELS, REVOLUTIONS, FPS, RADIUS,
        )

        # Command the first circle's start pose and stream (NOT saving) for
        # SETTLE_S so the arm settles off the home->start transit before capture.
        start_center = np.array([center_xy[0], center_xy[1], Z_LEVELS[0]], dtype=np.float64)
        start_quat = look_at_quat(start_center, AIM_POINT)
        start_action = {f"{ARM}_{k}": float(v)
                        for k, v in zip(EE_AXIS_KEYS, (*start_center, *start_quat))}
        logger.info("Settling on first pose for %.1fs before capture.", SETTLE_S)
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < SETTLE_S:
            robot.send_action(start_action)
            capturer.step(robot, save=False)
            time.sleep(1.0 / FPS)

        for z in Z_LEVELS:
            center = np.array([center_xy[0], center_xy[1], z], dtype=np.float64)
            logger.info("Circle at z=%.3f (center=%s).", z, center)
            trace_circle(robot, center, capturer)
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
    if capturer.image_size is not None and capturer.records:
        try:
            result = solve_handeye(capturer.records, capturer.image_size)
            if result is not None:
                data = load_results(WRIST_VIEW)
                data.update(result)
                save_results(WRIST_VIEW, data)
        except Exception:
            logger.exception("hand-eye solve failed")


if __name__ == "__main__":
    main()
