"""Optional LeRobot adapter for the standalone ARV driver."""

from franka_cameras.arv import ArvCamera as NativeArvCamera
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig  # noqa: F401 - compatibility export

from .config_arv import ArvCameraConfig  # noqa: F401 - compatibility export


class ArvCamera(NativeArvCamera, Camera):
    """Retain LeRobot's camera interface while sharing the native driver."""
