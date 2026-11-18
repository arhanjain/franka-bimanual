"""Small native camera contract, independent of LeRobot and its registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(kw_only=True)
class CameraConfig:
    fps: int | None = None
    width: int | None = None
    height: int | None = None


class Camera(ABC):
    def __init__(self, config: CameraConfig):
        self.fps = config.fps
        self.width = config.width
        self.height = config.height

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether capture has been started."""

    @abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """Start capture."""

    @abstractmethod
    def disconnect(self) -> None:
        """Stop capture and release SDK resources."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def __del__(self):
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            # Destructors may run after partial initialization or SDK shutdown.
            pass
