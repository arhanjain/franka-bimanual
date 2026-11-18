"""Native camera drivers; importing configs does not load either hardware SDK."""

from .config_arv import ArvCameraConfig
from .config_framos import FramosCameraConfig

__all__ = ["ArvCamera", "ArvCameraConfig", "FramosCamera", "FramosCameraConfig"]


def __getattr__(name):
    if name == "ArvCamera":
        from .arv import ArvCamera

        return ArvCamera
    if name == "FramosCamera":
        from .framos import FramosCamera

        return FramosCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
