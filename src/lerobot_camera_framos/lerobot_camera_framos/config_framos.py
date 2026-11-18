"""LeRobot registry entry for the standalone FRAMOS configuration."""

from dataclasses import dataclass

from franka_cameras.config_framos import FramosCameraConfig as NativeFramosCameraConfig
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("framos_camera")
@dataclass
class FramosCameraConfig(NativeFramosCameraConfig, CameraConfig):
    """Keep existing LeRobot config names and fields."""
