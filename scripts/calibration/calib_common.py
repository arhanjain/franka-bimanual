"""Shared helpers for the calibration scripts (camera_calibration.py + wrist
hand-eye). Chessboard geometry, corner detection, results JSON I/O, and small
rotation/transform utilities -- factored out so the two scripts can't drift.

Conventions:
- Chessboard PATTERN is (cols, rows) of INNER corners; SQUARE_SIZE in meters.
- Quaternions are stored WXYZ in results JSON (sim/diffik convention); scipy
  uses XYZW internally.
- results/<view>.json is the single per-view record (intrinsics + extrinsics +
  handeye blocks merged in place).
"""

import json
import os

import numpy as np
import cv2 as cv

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')  # results/<view>.json
SAMPLES_DIR = os.path.join(HERE, 'samples')  # samples/<view>/*.png (+ poses.json)

# chessboard geometry: inner-corner grid (cols, rows) and square size in meters.
# intrinsics are independent of square size; extrinsic distances scale with it.
PATTERN = (8, 13)
SQUARE_SIZE = 0.02

# corner sub-pixel refinement termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# --------------------------------------------------------------------------
# ChArUco board (shared by wrist hand-eye + scene-cam extrinsics so the two
# can't drift). 11x11 squares, DICT_4X4_1000, measured square/marker sizes.
# SQUARE size sets the metric scale; MARKER size only affects ArUco detection.
# --------------------------------------------------------------------------
CHARUCO_SQUARES = (11, 11)
CHARUCO_SQUARE_SIZE = 0.05    # 5 cm checker square (measured)
CHARUCO_MARKER_SIZE = 0.039   # 3.9 cm black ArUco marker (measured)
CHARUCO_MIN_CORNERS = 6       # min interpolated corners for a usable view

_charuco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
charuco_board = cv.aruco.CharucoBoard(
    CHARUCO_SQUARES, CHARUCO_SQUARE_SIZE, CHARUCO_MARKER_SIZE, _charuco_dict)
_charuco_detector = cv.aruco.CharucoDetector(charuco_board)

# Object-point origin shift: OpenCV's ChArUco object points start at a board
# CORNER; subtract this to move the frame origin to the board CENTER (== env
# origin), matching where the board is physically placed.
_CHARUCO_CENTER = np.array([0.5 * CHARUCO_SQUARES[0] * CHARUCO_SQUARE_SIZE,
                            0.5 * CHARUCO_SQUARES[1] * CHARUCO_SQUARE_SIZE, 0.0],
                           dtype=np.float32)

# Env re-aim: OpenCV's native ChArUco frame comes out (in env/physical
# directions) +X right / +Y back / +Z down. We want the env convention +X
# forward / +Y left / +Z up. Columns are the NEW axes written in the OLD frame:
#   new +X = forward = -(old +Y back)  -> ( 0,-1, 0)
#   new +Y = left    = -(old +X right) -> (-1, 0, 0)
#   new +Z = up      = -(old +Z down)  -> ( 0, 0,-1)
# det = +1 (PROPER rotation, NOT a reflection -- a single-axis negate would be a
# reflection and crash Rotation.from_matrix). VERIFIED via the wrist hand-eye.
BOARD_REAIM = np.array([[0.0, -1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0]], dtype=np.float64)


def detect_charuco(gray):
    """Detect ChArUco corners. Returns (corners Nx1x2, ids Nx1) or (None, None)."""
    corners, ids, _, _ = _charuco_detector.detectBoard(gray)
    if ids is None or len(ids) < CHARUCO_MIN_CORNERS:
        return None, None
    return corners, ids


def charuco_match_env(corners, ids):
    """Matched (obj_pts, img_pts) for solve/PnP, with obj_pts already expressed in
    the ENV frame (origin at board center, +X forward / +Y left / +Z up). Returns
    (None, None) if too few matched corners.

    obj_pts in env coords = (centered native points) @ BOARD_REAIM: for a row
    vector p_old, p_old @ M == (Mᵀ p_old) gives the same physical point in the
    re-aimed frame -- the exact equivalent of post-multiplying cam_T_board by M.
    """
    op, ip = charuco_board.matchImagePoints(corners, ids)
    if op is None or len(op) < CHARUCO_MIN_CORNERS:
        return None, None
    obj = op.reshape(-1, 3).astype(np.float32) - _CHARUCO_CENTER  # corner -> center
    obj = (obj @ BOARD_REAIM).astype(np.float32)                  # native -> env axes
    return obj, ip.reshape(-1, 2).astype(np.float32)


# --------------------------------------------------------------------------
# chessboard helpers
# --------------------------------------------------------------------------
def objp(pattern=PATTERN, square=SQUARE_SIZE) -> np.ndarray:
    """Board object points: (col,row,0)*square, origin at corner (0,0)."""
    pts = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    pts[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    pts[:, :2] *= square
    return pts


def find_corners(gray, pattern=PATTERN):
    """Detect + sub-pixel-refine chessboard corners. Returns (found, corners)."""
    found, corners = cv.findChessboardCorners(gray, pattern, None)
    if found:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


# --------------------------------------------------------------------------
# rotation / transform helpers
# --------------------------------------------------------------------------
def quat_wxyz_from_R(R) -> list:
    """3x3 rotation -> [w, x, y, z] (scipy returns xyzw)."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def T_from(p, R) -> np.ndarray:
    """4x4 homogeneous transform from position (3,) + rotation (3x3)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p).ravel()
    return T


def invT(T) -> np.ndarray:
    """Inverse of a 4x4 rigid transform."""
    Ti = np.eye(4)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def board_zup_flip(R_cam, t_cam):
    """Force the board frame's +Z up (keeping +X) for camera-in-board poses.

    A camera shooting a board from above must have positive height in the board
    frame; solvePnP's planar sign ambiguity (or a detected +Z facing down) can
    yield a Z-down frame. Flip 180 deg about board +X (keeps X, negates Y,Z -- a
    proper rotation) when t_cam[2] < 0. SCENE-extrinsics path only; do NOT apply
    in the hand-eye path, where cross-pose intrinsic-Z consistency is the signal.
    """
    if t_cam[2] < 0:
        flip = np.diag([1.0, -1.0, -1.0])
        return flip @ R_cam, flip @ t_cam
    return R_cam, t_cam


# --------------------------------------------------------------------------
# results JSON (one record per view; blocks merged in place)
# --------------------------------------------------------------------------
def results_path(view: str) -> str:
    return os.path.join(RESULTS_DIR, f'{view}.json')


def load_results(view: str) -> dict:
    path = results_path(view)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'view': view}


def save_results(view: str, data: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = results_path(view)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[results] wrote {path}')


# --------------------------------------------------------------------------
# captured samples (per-view image dir; shared by wrist + scene cam scripts)
# --------------------------------------------------------------------------
def samples_dir(view: str) -> str:
    """samples/<view>/ -- captured frames (+ poses.json) for that camera."""
    return os.path.join(SAMPLES_DIR, view)
