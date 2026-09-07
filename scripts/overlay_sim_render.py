#!/usr/bin/env python3
"""Overlay live real cameras on a reset-state render from simulation.

The default output of ``scripts/visualization/visualize_reset_states.sh`` is a
single composite PNG, rather than four independently named images.  This tool
understands that *default* layout and extracts each camera before matching it
by canonical view name to the real EnvFrameFranka camera stream:

    scene_left_0, scene_right_0, wrist_left_plus, wrist_right_minus

The window has three rows (SIM, REAL, OVERLAY) and one column per view.  The
live overlay blends the real image over the frozen simulated reference.  This
is intended to make residual camera-pose, intrinsics, and scene-model errors
visible while aiming or calibrating the real rig; it does not estimate a warp.

Run in the REAL venv (``third_party/franka-bimanual/.venv``):

  python scripts/overlay_sim_render.py \
      --sim-image /home/qirico/qirico/sim-improvement-aug26/experiments/dataset_generation/sim_render_9.png

  # Scene cameras plus only luigi's (right-arm) wrist camera:
  python scripts/overlay_sim_render.py --arms r --sim-image /path/to/sim_render_9.png

Keys: q/ESC quit; [ and ] decrease/increase real-image opacity; s save the
current grid to ``--out``.

The parser intentionally rejects non-default render layouts.  It expects the
four default cameras in this order, ``--tile-columns 2``, and the 34-pixel
header added by ``visualize_reset_states.py``.  Generate a default render when
using this tool.

Usage:
.venv/bin/python scripts/overlay_sim_render.py \
  --sim-image /home/qirico/qirico/sim-improvement-aug26/experiments/dataset_generation/sim_render_1.png
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tyro


# These names and this order are shared by the simulation renderer and the
# EnvFrameFranka default rig.  Never infer correspondence from camera-dict
# insertion order: the two stacks construct their dictionaries differently.
POLICY_VIEWS = (
    "scene_left_0",
    "scene_right_0",
    "wrist_left_plus",
    "wrist_right_minus",
)

# Default ``visualize_reset_states.py`` composite image geometry.
_HEADER_H = 34
_SCENE_W, _SCENE_H = 640, 480
_WRIST_W, _WRIST_H = 960, 600
_RENDER_W = 2 * _WRIST_W
_RENDER_H = _HEADER_H + _SCENE_H + _WRIST_H


@dataclass
class Args:
    sim_image: str
    """Path to a sim_render_N.png made with visualize_reset_states.sh defaults."""
    arms: Literal["l", "r", "lr"] = "lr"
    """Which arms' wrist cameras to connect (scene cameras are always connected)."""
    tile_h: int = 240
    """Display height of each camera cell in pixels; aspect ratio is retained."""
    real_alpha: float = 0.5
    """Initial opacity of the live real image over the simulated reference, in [0, 1]."""
    fps: float = 30.0
    """Maximum CV2 display refresh rate; camera reads may limit the actual rate."""
    out: str = "sim_real_overlay.png"
    """PNG written when s is pressed."""


def _to_uint8_rgb(frame: object) -> np.ndarray:
    """Convert an HWC real-camera frame to contiguous uint8 RGB."""
    try:
        import torch

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
    except ImportError:
        pass

    array = np.asarray(frame)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC RGB/RGBA image, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def load_default_reset_render(path: str) -> dict[str, np.ndarray]:
    """Load a default reset-render composite and return named RGB camera crops.

    ``_tile_frames`` pads the short 1280-pixel scene row to the 1920-pixel wrist
    row.  Therefore the expected geometry is:

        y=0:34       state header
        y=34:514     scene_left_0 | scene_right_0 | padding
        y=514:1114   wrist_left_plus | wrist_right_minus
    """
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"could not read image: {path}")
    if image_bgr.shape[:2] != (_RENDER_H, _RENDER_W):
        height, width = image_bgr.shape[:2]
        raise ValueError(
            f"{path} is {width}x{height}, but the default reset-render layout is "
            f"{_RENDER_W}x{_RENDER_H}. Regenerate with the default four camera "
            "names and --tile-columns 2."
        )

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    scene_y = _HEADER_H
    wrist_y = _HEADER_H + _SCENE_H
    crops = {
        "scene_left_0": image[scene_y : scene_y + _SCENE_H, 0:_SCENE_W],
        "scene_right_0": image[scene_y : scene_y + _SCENE_H, _SCENE_W : 2 * _SCENE_W],
        "wrist_left_plus": image[wrist_y : wrist_y + _WRIST_H, 0:_WRIST_W],
        "wrist_right_minus": image[wrist_y : wrist_y + _WRIST_H, _WRIST_W : 2 * _WRIST_W],
    }
    return {name: np.ascontiguousarray(frame) for name, frame in crops.items()}


def _make_cell(frame: np.ndarray | None, height: int, width: int, label: str, color: tuple[int, int, int]) -> np.ndarray:
    """Resize a frame for the display grid, with a labeled placeholder if absent."""
    if frame is None:
        cell = np.full((height, width, 3), 64, dtype=np.uint8)
        label = f"{label} (no frame)"
    else:
        cell = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(cell, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return cell


def _make_grid(
    sim: dict[str, np.ndarray], real: dict[str, np.ndarray | None], tile_h: int, real_alpha: float
) -> np.ndarray:
    """Build SIM / REAL / OVERLAY rows in RGB, preserving each view's aspect."""
    columns = []
    for view in POLICY_VIEWS:
        sim_frame = sim[view]
        real_frame = real.get(view)
        width = max(1, round(tile_h * sim_frame.shape[1] / sim_frame.shape[0]))

        sim_cell = _make_cell(sim_frame, tile_h, width, f"SIM  {view}", (0, 255, 0))
        real_cell = _make_cell(real_frame, tile_h, width, f"REAL {view}", (0, 255, 0))

        if real_frame is None:
            overlay = cv2.resize(sim_frame, (width, tile_h), interpolation=cv2.INTER_AREA)
        else:
            sim_base = cv2.resize(sim_frame, (width, tile_h), interpolation=cv2.INTER_AREA)
            real_base = cv2.resize(real_frame, (width, tile_h), interpolation=cv2.INTER_AREA)
            overlay = cv2.addWeighted(real_base, real_alpha, sim_base, 1.0 - real_alpha, 0.0)
        cv2.putText(
            overlay,
            f"OVERLAY real={real_alpha:.2f}",
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        columns.append(np.vstack((sim_cell, real_cell, overlay)))
    return np.hstack(columns)


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    if args.tile_h <= 0:
        raise SystemExit("--tile-h must be positive")
    if args.fps < 0:
        raise SystemExit("--fps must be non-negative")

    try:
        sim = load_default_reset_render(args.sim_image)
    except ValueError as error:
        raise SystemExit(f"[overlay_sim_render] {error}") from error

    from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig

    robot = EnvFrameFranka(
        EnvFrameFrankaConfig(
            active_arms=tuple(args.arms), enable_grippers=False, enable_cameras=True
        )
    )
    print(f"[overlay_sim_render] loaded {args.sim_image}")
    print(f"[overlay_sim_render] connecting cameras only (arms={args.arms}; arms may be offline) ...")
    robot.connect_cameras()

    alpha = float(np.clip(args.real_alpha, 0.0, 1.0))
    window = "sim / real / overlay  (q quit, [ ] opacity, s save)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"[overlay_sim_render] streaming {list(robot.cameras)}")
    period = 1.0 / args.fps if args.fps > 0 else 0.0

    try:
        while True:
            started = time.perf_counter()
            frames = robot.read_camera_frames()
            real: dict[str, np.ndarray | None] = {}
            for view in POLICY_VIEWS:
                frame = frames.get(view)
                if frame is None:
                    real[view] = None
                    continue
                try:
                    real[view] = _to_uint8_rgb(frame)
                except ValueError as error:
                    print(f"[overlay_sim_render] ignoring invalid {view} frame: {error}")
                    real[view] = None

            grid = _make_grid(sim, real, args.tile_h, alpha)
            cv2.imshow(window, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("["):
                alpha = max(0.0, alpha - 0.05)
            elif key == ord("]"):
                alpha = min(1.0, alpha + 0.05)
            elif key == ord("s"):
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                print(f"[overlay_sim_render] saved {out_path}")

            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[overlay_sim_render] done.")


if __name__ == "__main__":
    main()
