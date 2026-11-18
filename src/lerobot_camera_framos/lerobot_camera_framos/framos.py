"""Optional LeRobot adapter for the standalone FRAMOS driver."""

from franka_cameras.framos import FramosCamera as NativeFramosCamera
from lerobot.cameras.camera import Camera

from .config_framos import FramosCameraConfig  # noqa: F401 - compatibility export


class FramosCamera(NativeFramosCamera, Camera):
    """Retain LeRobot's camera interface while sharing the native driver."""
