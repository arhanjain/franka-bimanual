from dataclasses import dataclass

from .camera import CameraConfig


@dataclass
class ArvCameraConfig(CameraConfig):
    name: str = ""
    ip: str = ""
    pixel_format: str = "Mono8"
