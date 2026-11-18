"""LeRobot registry entry for the standalone ARV configuration."""

from dataclasses import dataclass

from franka_cameras.config_arv import ArvCameraConfig as NativeArvCameraConfig
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("arv_camera")
@dataclass
class ArvCameraConfig(NativeArvCameraConfig, CameraConfig):
    """Keep existing LeRobot config names and fields."""
