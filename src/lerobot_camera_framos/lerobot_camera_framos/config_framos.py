from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("framos_camera")
@dataclass
class FramosCameraConfig(CameraConfig):
    name: str = ""
    ip: str = ""
    serial_number: str = ""
    enable_color: bool = True
    enable_depth: bool = True
    color_width: int = 1280
    color_height: int = 720
    depth_width: int = 1280
    depth_height: int = 720
    align_to: str = "color"
    color_format: str = "rgb8"
    depth_format: str = "z16"
    #: librealsense only accepts discrete FPS (typically 6/15/30/60/90 on D415e).
    #: If unset, FPS is snapped from `CameraConfig.fps` automatically in `FramosCamera`.
    streaming_fps: int | None = None
    options: dict[str, float] = field(default_factory=dict)
    # FALLBACK ONLY. FramosCamera.connect() overwrites _intrinsics by reading the
    # negotiated color stream's factory intrinsics live, because the correct
    # matrix is both per-camera (the two D415e differ) and per-resolution (we run
    # 224x224 here, 640x480 in the envframe rig). These identity-ish placeholders
    # are only used if that live read fails; do NOT treat them as calibrated.
    intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    # D415e color is rectified (inverse_brown_conrady with zero coeffs), so there
    # is no lens distortion to model.
    distortion_coeffs: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    # TODO(extrinsics): camera-in-world pose, used by get_depth() to lift the
    # point cloud into the env frame. These are STALE -- carried over from the old
    # BFS@1280x720 calibration and never re-measured for the FRAMOS cams (and are
    # a single shared default though the two D415e sit at different poses). Left
    # in place to avoid silently changing get_depth() behavior; recapture with
    # scripts/calibration/camera_calibration.py extrinsics and set per camera.
    r_cam_in_world: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (-0.91580561, -0.17701555, 0.36050739),
        (-0.34811877, 0.797506, -0.49274486),
        (-0.2002833, -0.5767579, -0.79198291),
    )
    t_cam_in_world: tuple[float, float, float] = (-0.2808756, 0.38170682, 0.64699288)
