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
CHARUCO_MIN_CORNERS = 10       # min interpolated corners for a usable view

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


# --------------------------------------------------------------------------
# OpenCV fisheye -> Isaac Sim legacy f-theta camera conversion
# --------------------------------------------------------------------------
_FISHEYE_POLYNOMIAL_SOURCE = "opencv_fisheye_monotonic_full_sensor_fit_v2"
_FISHEYE_FIT_MONOTONIC_MARGIN = 0.95
# Outside the observed/source-model range there is no trustworthy target image
# geometry. Keep the full rectangular sensor below a 180-degree diagonal FOV,
# while allowing up to 25 degrees of smooth extension beyond the source fit.
_FISHEYE_FULL_SENSOR_MAX_ANGLE = np.deg2rad(85.0)
_FISHEYE_FULL_SENSOR_MAX_EXTENSION = np.deg2rad(25.0)
_FISHEYE_MIN_NORMALIZED_DTHETA_DR = 1.0e-3


def _opencv_fisheye_theta_distortion(theta: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """Return OpenCV fisheye's distorted angle for an undistorted ray angle."""
    k1, k2, k3, k4 = distortion
    theta2 = theta * theta
    return theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))


def _opencv_fisheye_monotonic_angle(distortion: np.ndarray) -> float:
    """First positive angle where the OpenCV fisheye radial map stops increasing.

    The OpenCV model is only safely invertible before this point. If its
    derivative has no positive root before 90 degrees, use 90 degrees as the
    practical limit for a forward-looking wrist camera.
    """
    k1, k2, k3, k4 = distortion
    # d(theta_d)/d(theta), expressed in s = theta**2.
    roots = np.roots([9.0 * k4, 7.0 * k3, 5.0 * k2, 3.0 * k1, 1.0])
    positive_angles = [
        np.sqrt(float(root.real))
        for root in roots
        if abs(root.imag) < 1.0e-9 and root.real > 0.0
    ]
    return float(min([np.pi / 2.0, *positive_angles]))


def _observed_fisheye_max_angle(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    """Largest off-axis ray angle represented by observed calibration points."""
    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if len(points) == 0:
        raise ValueError("image_points must contain at least one pixel")
    undistorted = cv.fisheye.undistortPoints(points, camera_matrix, distortion)
    tangent_radius = np.linalg.norm(undistorted.reshape(-1, 2), axis=1)
    return float(np.max(np.arctan(tangent_radius)))


def fit_opencv_fisheye_to_isaac_sim_polynomial(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    *,
    image_points: np.ndarray | None = None,
    max_fov_deg: float | None = None,
    theta_samples: int = 256,
    azimuth_samples: int = 32,
) -> dict:
    """Fit OpenCV fisheye data to Isaac Sim's legacy f-theta model.

    OpenCV maps ray angle to pixel radius with a theta polynomial. Isaac Sim's
    ``fisheyePolynomial`` maps pixel radius back to angle. This function fits
    that inverse map, constraining its constant term to zero so the optical
    axis remains at the configured principal point.

    ``image_points`` should be the ChArUco detections used by calibration. It
    limits the *source* part of an automatic fit to the observed field of
    view. With no points (the standalone backfill use case), that source part
    uses 95% of the OpenCV model's monotonic range instead. The source fit is
    then extended to the furthest rectangular-sensor corner with a conservative
    monotonic cubic target, and the final fifth-order polynomial is constrained
    to stay one-to-one over that entire sensor. This avoids extrapolating an
    inner-field polynomial across the top/bottom of the rendered image.

    ``max_fov_deg`` explicitly selects a smaller source field of view and is
    rejected if it reaches the non-invertible range.
    """
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {matrix.shape}")
    if coeffs.shape != (4,):
        raise ValueError(f"OpenCV fisheye distortion must contain k1..k4, got {coeffs.shape}")
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must contain positive width and height, got {image_size}")
    if theta_samples < 8 or azimuth_samples < 4:
        raise ValueError("theta_samples must be >= 8 and azimuth_samples must be >= 4")

    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"camera focal lengths must be positive, got fx={fx}, fy={fy}")

    monotonic_angle = _opencv_fisheye_monotonic_angle(coeffs)
    safe_monotonic_angle = monotonic_angle * _FISHEYE_FIT_MONOTONIC_MARGIN
    observed_max_angle = None
    if image_points is not None:
        observed_max_angle = _observed_fisheye_max_angle(image_points, matrix, coeffs)

    if max_fov_deg is not None:
        if max_fov_deg <= 0.0:
            raise ValueError(f"max_fov_deg must be positive, got {max_fov_deg}")
        theta_limit = float(np.deg2rad(max_fov_deg) / 2.0)
        if theta_limit > safe_monotonic_angle:
            raise ValueError(
                f"requested max_fov_deg={max_fov_deg:.3f} exceeds the safe monotonic "
                f"limit of {np.rad2deg(2.0 * safe_monotonic_angle):.3f} degrees"
            )
        fit_domain_source = "explicit_max_fov_deg"
    elif observed_max_angle is not None:
        theta_limit = min(observed_max_angle, safe_monotonic_angle)
        fit_domain_source = "observed_calibration_points"
    else:
        theta_limit = safe_monotonic_angle
        fit_domain_source = "source_monotonic_limit_fallback"

    if theta_limit <= 0.0:
        raise ValueError("fitting domain has no positive ray angle")

    # Sample the source model evenly in angle and azimuth. Sampling azimuths
    # turns OpenCV's separate fx/fy into the least-squares radial f-theta fit.
    theta = np.linspace(0.0, theta_limit, theta_samples)
    azimuth = np.linspace(0.0, 2.0 * np.pi, azimuth_samples, endpoint=False)
    theta_grid, azimuth_grid = np.meshgrid(theta, azimuth, indexing="ij")
    theta_distorted = _opencv_fisheye_theta_distortion(theta_grid, coeffs)
    pixel_radius = theta_distorted * np.sqrt(
        (fx * np.cos(azimuth_grid)) ** 2 + (fy * np.sin(azimuth_grid)) ** 2
    )

    sensor_radius = float(max(
        np.hypot(pixel_x - cx, pixel_y - cy)
        for pixel_x in (0.0, float(width))
        for pixel_y in (0.0, float(height))
    ))
    # A requested/observed source range can extend beyond the rectangular
    # sensor. Only retain samples that can actually appear in the render.
    source_mask = pixel_radius.reshape(-1) <= sensor_radius
    radius = pixel_radius.reshape(-1)[source_mask]
    angles = theta_grid.reshape(-1)[source_mask]
    source_theta_distorted = theta_distorted.reshape(-1)[source_mask]
    source_azimuth = azimuth_grid.reshape(-1)[source_mask]
    source_radius = float(np.max(radius))
    source_radius_fraction = source_radius / sensor_radius
    normalized_radius = radius / sensor_radius
    source_design = np.column_stack([normalized_radius**order for order in range(1, 6)])
    source_nonzero, _, _, _ = np.linalg.lstsq(source_design, angles, rcond=None)
    source_polynomial = np.array([0.0, *source_nonzero], dtype=np.float64)

    # The lens is only calibrated inside ``source_radius``. Rather than apply
    # its degree-five approximation outside that range, add a gentle, monotonic
    # Hermite extension to the farthest sensor corner. The extension is a
    # regularizer: it has no claim of being measured lens geometry.
    boundary_angle = float(np.polynomial.polynomial.polyval(source_radius_fraction, source_polynomial))
    boundary_slope = float(sum(
        order * source_polynomial[order] * source_radius_fraction ** (order - 1)
        for order in range(1, len(source_polynomial))
    ))
    boundary_slope = max(boundary_slope, _FISHEYE_MIN_NORMALIZED_DTHETA_DR)
    if source_radius_fraction >= 1.0 - 1.0e-6:
        # The source range already covers the sensor, so no extrapolation is
        # necessary. Keeping an empty extension also avoids a zero-width
        # Hermite interval at an exactly covered corner.
        extension_angle = boundary_angle
        extension_targets = np.empty(0, dtype=np.float64)
        extension_design = np.empty((0, 5), dtype=np.float64)
        extension_kind = "not_needed_source_covers_sensor"
    else:
        extension_angle = min(
            _FISHEYE_FULL_SENSOR_MAX_ANGLE,
            theta_limit + _FISHEYE_FULL_SENSOR_MAX_EXTENSION,
        )
        extension_angle = max(
            extension_angle,
            boundary_angle + _FISHEYE_MIN_NORMALIZED_DTHETA_DR * (1.0 - source_radius_fraction),
        )
        extension_radius = np.linspace(source_radius_fraction, 1.0, theta_samples)
        extension_t = (extension_radius - source_radius_fraction) / (1.0 - source_radius_fraction)
        extension_secant = (extension_angle - boundary_angle) / (1.0 - source_radius_fraction)
        extension_start_slope = min(boundary_slope, 2.5 * extension_secant)
        extension_end_slope = min(
            max(0.1 * boundary_slope, _FISHEYE_MIN_NORMALIZED_DTHETA_DR),
            2.5 * extension_secant,
        )
        h00 = 2.0 * extension_t**3 - 3.0 * extension_t**2 + 1.0
        h10 = extension_t**3 - 2.0 * extension_t**2 + extension_t
        h01 = -2.0 * extension_t**3 + 3.0 * extension_t**2
        h11 = extension_t**3 - extension_t**2
        extension_targets = (
            h00 * boundary_angle
            + h10 * (1.0 - source_radius_fraction) * extension_start_slope
            + h01 * extension_angle
            + h11 * (1.0 - source_radius_fraction) * extension_end_slope
        )
        extension_design = np.column_stack([extension_radius**order for order in range(1, 6)])
        extension_kind = "monotonic_cubic_hermite_regularizer"

    design = np.concatenate([source_design, extension_design])
    targets = np.concatenate([angles, extension_targets])
    # Each source sample (angle, azimuth) and each extension sample contributes
    # equally. There are many more source samples, so calibrated geometry stays
    # dominant while the extension prevents peripheral extrapolation artifacts.
    weights = np.ones(len(targets), dtype=np.float64)
    fit_nonzero, _, _, _ = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]), targets * np.sqrt(weights), rcond=None
    )

    # Enforce positive angular slope over the whole physical sensor. In normal
    # cases the least-squares solution already satisfies this. If it does not,
    # solve the same normalized least-squares objective with dense derivative
    # constraints; scipy is already a dependency of the calibration scripts.
    validation_normalized_radius = np.linspace(0.0, 1.0, 4097)
    derivative_design = np.column_stack([
        order * validation_normalized_radius ** (order - 1) for order in range(1, 6)
    ])
    if np.min(derivative_design @ fit_nonzero) < _FISHEYE_MIN_NORMALIZED_DTHETA_DR:
        from scipy.optimize import minimize

        weight_sum = float(np.sum(weights))

        def objective(coefficients):
            residual = design @ coefficients - targets
            return float(np.sum(weights * residual**2) / weight_sum)

        def objective_jacobian(coefficients):
            residual = design @ coefficients - targets
            return 2.0 * (design.T @ (weights * residual)) / weight_sum

        initial = np.zeros(5, dtype=np.float64)
        initial[0] = max(extension_angle, _FISHEYE_MIN_NORMALIZED_DTHETA_DR)
        constrained = minimize(
            objective,
            initial,
            jac=objective_jacobian,
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": lambda coefficients: derivative_design @ coefficients - _FISHEYE_MIN_NORMALIZED_DTHETA_DR,
                "jac": lambda coefficients: derivative_design,
            },
            options={"ftol": 1.0e-12, "maxiter": 2000},
        )
        if not constrained.success:
            raise ValueError(f"full-sensor f-theta fit failed: {constrained.message}")
        fit_nonzero = constrained.x

    normalized_polynomial = np.array([0.0, *fit_nonzero], dtype=np.float64)
    polynomial = np.array(
        [0.0, *[fit_nonzero[order - 1] / sensor_radius**order for order in range(1, 6)]],
        dtype=np.float64,
    )
    min_derivative = float(np.min(derivative_design @ fit_nonzero) / sensor_radius)
    if min_derivative <= 0.0:
        raise ValueError("full-sensor f-theta polynomial is not monotonic")

    # Report the source-domain projection error in pixels, not just angular fit
    # error. The inverse is found by interpolation because the final fit is
    # monotonic from the optical centre to the furthest sensor corner.
    validation_radius = np.linspace(0.0, sensor_radius, 65536)
    fitted_angles = np.polynomial.polynomial.polyval(
        validation_radius / sensor_radius, normalized_polynomial
    )
    target_radius = np.interp(angles, fitted_angles, validation_radius)
    source_u = cx + fx * source_theta_distorted * np.cos(source_azimuth)
    source_v = cy + fy * source_theta_distorted * np.sin(source_azimuth)
    target_u = cx + target_radius * np.cos(source_azimuth)
    target_v = cy + target_radius * np.sin(source_azimuth)
    pixel_error = np.hypot(target_u - source_u, target_v - source_v)

    result = {
        "source": _FISHEYE_POLYNOMIAL_SOURCE,
        "projection_type": "fisheyePolynomial",
        "fisheye_nominal_width": float(width),
        "fisheye_nominal_height": float(height),
        "fisheye_optical_centre_x": float(cx),
        "fisheye_optical_centre_y": float(cy),
        "fisheye_max_fov": float(np.rad2deg(2.0 * fitted_angles[-1])),
        "fisheye_polynomial_a": float(polynomial[0]),
        "fisheye_polynomial_b": float(polynomial[1]),
        "fisheye_polynomial_c": float(polynomial[2]),
        "fisheye_polynomial_d": float(polynomial[3]),
        "fisheye_polynomial_e": float(polynomial[4]),
        "fisheye_polynomial_f": float(polynomial[5]),
        "fit_domain_source": fit_domain_source,
        "fit_domain_max_angle_deg": float(np.rad2deg(theta_limit)),
        "source_monotonic_max_angle_deg": float(np.rad2deg(monotonic_angle)),
        "fit_rms_px": float(np.sqrt(np.mean(pixel_error**2))),
        "fit_max_px": float(np.max(pixel_error)),
        "fit_min_dtheta_dr": min_derivative,
        "fit_sample_count": int(radius.size),
        "fit_source_max_radius_px": source_radius,
        "fit_source_max_radius_fraction": source_radius_fraction,
        "full_sensor_extension": extension_kind,
        "full_sensor_target_corner_angle_deg": float(np.rad2deg(extension_angle)),
        "full_sensor_corner_angle_deg": float(np.rad2deg(fitted_angles[-1])),
        "full_sensor_max_radius_px": sensor_radius,
        "full_sensor_min_dtheta_dr": min_derivative,
        "full_sensor_fit_sample_count": int(len(targets)),
    }
    if observed_max_angle is not None:
        result["observed_max_angle_deg"] = float(np.rad2deg(observed_max_angle))
    return result
