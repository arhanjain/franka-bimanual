"""Scene-camera calibration, sourced from the EnvFrameFranka rig.

Calibrates the two fixed FRAMOS scene cameras (scene_left_0, scene_right_0):
intrinsics (read live off the RealSense device) and extrinsics (camera pose in
the env frame, via a chessboard at the env origin). Cameras are built from
EnvFrameFrankaConfig's default rig so frames come at the exact policy resolution
(640x480) and processing path; only the requested camera is connected -- the
arms are NOT brought up, so this runs regardless of NUC/arm state.

For the WRIST cameras (ARV, mounted on the arms) use
scripts/calibration/wrist_handeye_calibration.py, which moves the arm to get
pose diversity and recovers intrinsics + the base-in-env transform together.

Usage (pick a side, like wrist_handeye.py):
  python camera_calibration.py --side r     # calibrates scene_right_0
  python camera_calibration.py --side l     # calibrates scene_left_0

One run does it all for that camera: read factory intrinsics, auto-capture a few
board-detected frames a second apart (board flat at the env origin; the stream is
shown but no manual shutter -- cam + board are fixed), then solve + save the
camera-in-env extrinsics. Results merge into results/<view>.json. Needs the rig
venv (third_party/franka-bimanual/.venv).

Capture keys: SPACE/c save, u undo last, q/ESC stop.

Board (extrinsics): the SAME 11x11 ChArUco board + env frame as the wrist
hand-eye (calib_common.charuco_*). Place the board flat at the env origin; the
shared `charuco_match_env` expresses object points in the env frame (origin at
the board CENTER, +X forward / +Y left / +Z up), so solvePnP yields env->cam
directly and the recovered camera pose is camera-in-env -- no per-detection
+Z-up flip needed. This is the identical env convention the wrist calibration
uses, so both land in one consistent world frame for plot_world_frame.py.
"""

import glob
import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import cv2 as cv
import tyro

from calib_common import (
    charuco_match_env, detect_charuco,
    load_results, results_path, samples_dir, save_results,
)

# Side flag -> scene view name (matches the wrist_handeye.py --arm convention:
# r = RIGHT = luigi side, l = LEFT = mario side).
VIEW_BY_SIDE = {'r': 'scene_right_0', 'l': 'scene_left_0'}

# Auto-capture: the scene cam + board are both fixed, so just grab a few
# board-detected frames a second apart (no manual shutter).
N_CAPTURES = 5
CAPTURE_INTERVAL_S = 1.0


# --------------------------------------------------------------------------
# live camera (built from the EnvFrameFranka rig; cameras only, no arms)
# --------------------------------------------------------------------------
def _open_view_camera(view: str):
    """Build the default EnvFrameFranka rig and connect ONLY `view`'s camera.

    Returns (robot, camera). The arms are never connected, so this works with
    the NUCs offline. Caller must robot.cameras[view].disconnect()."""
    from lerobot_robot_envframe_franka import EnvFrameFrankaConfig, EnvFrameFranka

    cfg = EnvFrameFrankaConfig(active_arms=('l', 'r'), enable_grippers=False, enable_cameras=True)
    robot = EnvFrameFranka(cfg)
    if view not in robot.cameras:
        raise SystemExit(f'Unknown view {view!r}. Available: {sorted(robot.cameras)}')
    cam = robot.cameras[view]
    print(f'[cam] connecting {view} ...')
    cam.connect()
    return robot, cam


# --------------------------------------------------------------------------
# capture: auto-grab N board-detected frames a second apart (no manual shutter;
# the scene cam + board are both fixed). Shows the live stream while it works.
# --------------------------------------------------------------------------
def capture(view: str, cam) -> None:
    out_dir = samples_dir(view)
    os.makedirs(out_dir, exist_ok=True)
    # Start clean so a re-run never mixes stale frames into the solve.
    for f in glob.glob(os.path.join(out_dir, '*.png')):
        os.remove(f)

    win = f'{view} -- auto-capturing {N_CAPTURES} frames'
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    saved = 0
    last_save = 0.0
    print(f'[capture] auto-capturing {N_CAPTURES} board frames into {out_dir}')
    try:
        while saved < N_CAPTURES:
            rgb = cam.read()
            bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
            corners, ids = detect_charuco(gray)
            found = corners is not None

            now = time.perf_counter()
            if found and now - last_save >= CAPTURE_INTERVAL_S:
                saved += 1
                cv.imwrite(os.path.join(out_dir, f'{saved}.png'), bgr)  # clean frame
                last_save = now
                print(f'[capture] saved {saved}/{N_CAPTURES}')

            view_img = bgr.copy()
            if found:
                cv.aruco.drawDetectedCornersCharuco(view_img, corners, ids)
            n_c = 0 if not found else len(ids)
            color = (0, 255, 0) if found else (0, 0, 255)
            cv.putText(view_img, f'{f"DETECTED ({n_c})" if found else "no board"}  '
                       f'saved={saved}/{N_CAPTURES}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)
            cv.imshow(win, view_img)
            if (cv.waitKey(1) & 0xFF) in (ord('q'), 27):  # bail early if needed
                break
    finally:
        cv.destroyAllWindows()
    print(f'[capture] done -- {saved} image(s) in {out_dir}')


# --------------------------------------------------------------------------
# subcommand: framos-intrinsics (read on-device factory intrinsics)
# --------------------------------------------------------------------------
def framos_intrinsics(view: str, cam) -> None:
    import pyrealsense2 as rs

    profile = getattr(cam, '_profile', None)
    if profile is None:
        raise SystemExit(f'{view}: camera has no RealSense profile after connect')
    vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = vsp.get_intrinsics()
    mtx = [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]
    dist = list(intr.coeffs)  # [k1,k2,p1,p2,k3] (Brown-Conrady)
    print(f'[framos] {view}  {intr.width}x{intr.height}  model={intr.model}')
    print('  matrix:', mtx)
    print('  distortion:', dist)

    data = load_results(view)
    data['image_size'] = [int(intr.width), int(intr.height)]
    data['intrinsics'] = {
        'source': 'factory',
        'matrix': mtx,
        'distortion': dist,
        'model': str(intr.model),
    }
    save_results(view, data)


# --------------------------------------------------------------------------
# subcommand: extrinsics (camera-in-env via solvePnP; board frame == env frame)
# --------------------------------------------------------------------------
def extrinsics(view: str) -> None:
    from scipy.spatial.transform import Rotation

    data = load_results(view)
    intr = data.get('intrinsics')
    if not intr:
        raise SystemExit(f'No intrinsics in {results_path(view)}.')
    mtx = np.array(intr['matrix'], dtype=np.float64)
    dist = np.array(intr['distortion'], dtype=np.float64)

    img_dir = samples_dir(view)
    images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if not images:
        raise SystemExit(f'No images in {img_dir}.')

    translations, quats, errs = [], [], []
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners, ids = detect_charuco(gray)
        if corners is None:
            print(f'[extrinsics] no board in {os.path.basename(fname)} -- skipped')
            continue
        # obj points already in the ENV frame (origin = board center, +X fwd /
        # +Y left / +Z up), so solvePnP's (rvec,tvec) is env->cam directly. No
        # +Z-up flip hack needed -- the env frame is fixed by the board placement.
        obj, ip = charuco_match_env(corners, ids)
        if obj is None:
            print(f'[extrinsics] too few corners in {os.path.basename(fname)} -- skipped')
            continue
        ok, rvec, tvec = cv.solvePnP(obj, ip, mtx, dist)
        if not ok:
            print(f'[extrinsics] solvePnP failed on {os.path.basename(fname)} -- skipped')
            continue
        proj, _ = cv.projectPoints(obj, rvec, tvec, mtx, dist)
        # RMS reproj error; reshape both to (N,2) so shapes/dtypes match.
        d = proj.reshape(-1, 2) - ip.reshape(-1, 2)
        errs.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
        R, _ = cv.Rodrigues(rvec)
        # env->camera (R,t); camera pose in the env frame is the inverse.
        R_cam = R.T
        t_cam = (-R.T @ tvec).ravel()
        translations.append(t_cam)
        quats.append(Rotation.from_matrix(R_cam).as_quat())  # xyzw
    if not translations:
        raise SystemExit(f'No usable board detections in {img_dir}.')

    t_mean = np.mean(translations, axis=0)
    # average quaternions (sign-aligned to the first), then renormalize
    q = np.array(quats)
    q[np.sum(q * q[0], axis=1) < 0] *= -1
    q_mean = q.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    R_mean = Rotation.from_quat(q_mean).as_matrix()
    qx, qy, qz, qw = q_mean

    print(f'[extrinsics] {len(translations)} image(s)  mean reproj {np.mean(errs):.3f} px')
    print('  camera-in-env translation (m):', t_mean)
    print('  camera-in-env quaternion wxyz:', [qw, qx, qy, qz])

    data['extrinsics'] = {
        'frame': 'env',
        'translation_xyz': t_mean.tolist(),
        'quaternion_wxyz': [float(qw), float(qx), float(qy), float(qz)],
        'rotation_matrix': R_mean.tolist(),
        'n_images': len(translations),
        'reproj_error_px': float(np.mean(errs)),
    }
    save_results(view, data)


@dataclass
class Args:
    side: Literal['l', 'r'] = 'r'
    """which scene camera to calibrate: r = scene_right_0, l = scene_left_0."""


def main() -> None:
    args = tyro.cli(Args)
    view = VIEW_BY_SIDE[args.side]
    print(f'[calib] {args.side} -> {view}')
    # Open the camera ONCE and share it across intrinsics + capture (no double
    # connect); extrinsics reads the saved frames from disk, so no camera needed.
    _, cam = _open_view_camera(view)
    try:
        framos_intrinsics(view, cam)   # 1. factory intrinsics
        capture(view, cam)             # 2. auto-capture board frames (board at env origin)
    finally:
        cam.disconnect()
    extrinsics(view)                   # 3. solve + save camera-in-env


if __name__ == '__main__':
    main()
