"""Hardware-free checks for native capture and optional LeRobot adapters."""

from __future__ import annotations

import importlib
import importlib.abc
import subprocess
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))


@pytest.fixture
def drivers(monkeypatch):
    gi = ModuleType("gi")
    gi.require_version = lambda *_: None
    repository = ModuleType("gi.repository")
    repository.Aravis = SimpleNamespace(BufferStatus=SimpleNamespace(SUCCESS=0))
    rs = ModuleType("pyrealsense2")
    rs.format = SimpleNamespace(
        **{name: name for name in ("bgr8", "rgb8", "yuyv", "z16", "y8", "y16")}
    )
    rs.stream = SimpleNamespace(color="color", depth="depth", infrared="infrared")
    for name, module in (
        ("gi", gi),
        ("gi.repository", repository),
        ("pyrealsense2", rs),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # Each test gets SDK stubs without leaking them into other test modules.
    for name in ("franka_cameras.arv", "franka_cameras.framos"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    class BlockML(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.split(".")[0] in {"lerobot", "torch", "transformers", "open3d"}:
                raise AssertionError(f"unexpected driver dependency: {fullname}")

    monkeypatch.setattr(sys, "meta_path", [BlockML(), *sys.meta_path])
    arv = importlib.import_module("franka_cameras.arv")
    framos = importlib.import_module("franka_cameras.framos")
    yield arv, framos
    for name in ("franka_cameras.arv", "franka_cameras.framos"):
        sys.modules.pop(name, None)


def test_config_import_needs_no_sdk_or_lerobot():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'lerobot', 'torch', 'transformers', 'open3d', 'gi', 'pyrealsense2'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from franka_cameras import ArvCameraConfig, FramosCameraConfig
assert ArvCameraConfig(fps=90).fps == 90
assert FramosCameraConfig().enable_depth
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, " + repr(str(SOURCE)) + ");" + code,
        ],
        check=True,
    )


def test_missing_sdk_error_is_not_hidden():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'gi':
            raise ImportError('Aravis SDK missing')
sys.meta_path.insert(0, Block())
try:
    from franka_cameras import ArvCamera
except ImportError as exc:
    assert 'Aravis SDK missing' in str(exc)
else:
    raise AssertionError('driver import must fail visibly')
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, " + repr(str(SOURCE)) + ");" + code,
        ],
        check=True,
    )


class Buffer:
    def __init__(self, value, *, status=0):
        self.value = value
        self.status = status

    def get_status(self):
        return self.status


class Stream:
    def __init__(self, *buffers):
        self.ready = deque(buffers)
        self.recycled = []

    def timeout_pop_buffer(self, timeout):
        return self.try_pop_buffer()

    def try_pop_buffer(self):
        return self.ready.popleft() if self.ready else None

    def push_buffer(self, buffer):
        self.recycled.append(buffer)


def test_aravis_reads_newest_and_preserves_stale_sequence(drivers):
    arv, _ = drivers
    camera = arv.ArvCamera(arv.ArvCameraConfig(width=2, height=1, fps=90))
    camera._camera = SimpleNamespace(stop_acquisition=lambda: None)
    camera._stream = Stream(Buffer(1), Buffer(2), Buffer(3))
    camera._buffer_to_rgb = lambda buffer: np.full(
        (1, 2, 3), buffer.value, dtype=np.uint8
    )
    frame = camera.async_read(10)
    np.testing.assert_array_equal(frame, 3)
    assert len(camera._stream.recycled) == 3
    assert camera.frame_sequence == 1
    timestamp = camera.last_frame_monotonic_s
    assert timestamp > 0
    frame[:] = 99
    np.testing.assert_array_equal(camera.async_read(0), 3)
    assert camera.frame_sequence == 1
    assert camera.last_frame_monotonic_s == timestamp
    camera._stream.ready.append(Buffer(4, status=1))
    np.testing.assert_array_equal(camera.async_read(0), 3)
    assert camera.frame_sequence == 1
    assert len(camera._stream.recycled) == 4
    camera.disconnect()
    assert not camera.is_connected


def test_aravis_blank_is_never_fresh(drivers):
    arv, _ = drivers
    camera = arv.ArvCamera(arv.ArvCameraConfig(width=2, height=1))
    camera._stream = Stream()
    np.testing.assert_array_equal(
        camera.async_read(0), np.zeros((1, 2, 3), dtype=np.uint8)
    )
    assert camera.frame_sequence == 0
    assert camera.last_frame_monotonic_s == 0
    with pytest.raises(TimeoutError):
        camera.read()
    camera.disconnect()


def test_aravis_recycles_buffer_when_decode_fails(drivers):
    arv, _ = drivers
    camera = arv.ArvCamera(arv.ArvCameraConfig(width=2, height=1))
    camera._stream = Stream(Buffer(1))

    def fail(_):
        raise ValueError("truncated payload")

    camera._buffer_to_rgb = fail
    camera.async_read(0)
    assert len(camera._stream.recycled) == 1
    assert camera.frame_sequence == 0
    camera.disconnect()


class Pipeline:
    def __init__(self, image):
        self.image = image
        self.stopped = False

    def wait_for_frames(self, **_):
        if self.image is None:
            raise RuntimeError("timeout")
        return SimpleNamespace(
            get_color_frame=lambda: SimpleNamespace(get_data=lambda: self.image),
            get_depth_frame=lambda: None,
        )

    def stop(self):
        self.stopped = True


def test_framos_rgb_and_stale_sequence(drivers):
    _, framos = drivers
    camera = framos.FramosCamera(
        framos.FramosCameraConfig(
            width=2, height=1, color_format="bgr8", enable_depth=False
        )
    )
    pipeline = Pipeline(np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8))
    camera._pipeline = pipeline
    camera._profile = object()
    frame = camera.async_read(0)
    np.testing.assert_array_equal(frame, [[[3, 2, 1], [6, 5, 4]]])
    assert frame.flags.c_contiguous
    assert camera.frame_sequence == 1
    timestamp = camera.last_frame_monotonic_s
    pipeline.image = None
    frame[:] = 99
    np.testing.assert_array_equal(camera.async_read(0), [[[3, 2, 1], [6, 5, 4]]])
    assert camera.frame_sequence == 1
    assert camera.last_frame_monotonic_s == timestamp > 0
    camera.disconnect()
    assert pipeline.stopped
    assert not camera.is_connected


def test_framos_start_failure_stops_pipeline(drivers, monkeypatch):
    _, framos = drivers
    pipeline = Pipeline(None)

    def fail(_):
        raise RuntimeError("unsupported mode")

    pipeline.start = fail
    monkeypatch.setattr(framos.rs, "pipeline", lambda: pipeline, raising=False)
    monkeypatch.setattr(
        framos.rs,
        "config",
        lambda: SimpleNamespace(enable_stream=lambda *_: None),
        raising=False,
    )
    camera = framos.FramosCamera(framos.FramosCameraConfig(enable_depth=False))
    with pytest.raises(RuntimeError, match="pipeline.start failed: unsupported mode"):
        camera.connect()
    assert pipeline.stopped
    assert not camera.is_connected


def test_context_cleanup_on_exception(drivers):
    arv, _ = drivers
    camera = arv.ArvCamera(arv.ArvCameraConfig())
    camera.connect = lambda: setattr(
        camera, "_camera", SimpleNamespace(stop_acquisition=lambda: None)
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        with camera as opened:
            assert opened is camera
            raise RuntimeError("capture failed")
    assert camera._camera is None


@pytest.mark.parametrize(
    "kind,class_name,registry_name",
    [
        ("arv", "ArvCamera", "arv_camera"),
        ("framos", "FramosCamera", "framos_camera"),
    ],
)
def test_optional_lerobot_adapters_keep_registry_and_native_defaults(
    drivers, monkeypatch, kind, class_name, registry_name
):
    from franka_cameras.camera import Camera as NativeCamera
    from franka_cameras.camera import CameraConfig as NativeConfig

    # Reproduce only LeRobot's base-class/registry contract; no ML stack needed.
    @dataclass(kw_only=True)
    class LeRobotConfig:
        fps: int | None = None
        width: int | None = None
        height: int | None = None

        @classmethod
        def register_subclass(cls, name):
            def register(subclass):
                registry[name] = subclass
                return subclass

            return register

    registry = {}
    configs = ModuleType("lerobot.cameras.configs")
    configs.CameraConfig = LeRobotConfig
    cameras = ModuleType("lerobot.cameras.camera")
    cameras.Camera = type("LeRobotCamera", (), {})
    monkeypatch.setitem(sys.modules, "lerobot.cameras.configs", configs)
    monkeypatch.setitem(sys.modules, "lerobot.cameras.camera", cameras)
    plugin = f"lerobot_camera_{kind}"
    monkeypatch.syspath_prepend(str(SOURCE.parent / plugin))
    for name in (plugin, f"{plugin}.{kind}", f"{plugin}.config_{kind}"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(plugin)
    config = getattr(module, class_name + "Config")(width=640, height=480, fps=60)
    native_config = getattr(
        importlib.import_module("franka_cameras"), class_name + "Config"
    )(width=640, height=480, fps=60)
    assert isinstance(config, NativeConfig)
    assert isinstance(config, LeRobotConfig)
    assert asdict(config) == asdict(native_config)
    assert registry[registry_name] is type(config)
    camera = getattr(module, class_name)(config)
    assert isinstance(camera, NativeCamera)
    assert isinstance(camera, cameras.Camera)
    assert camera.fps == 60
    for name in (plugin, f"{plugin}.{kind}", f"{plugin}.config_{kind}"):
        sys.modules.pop(name, None)
