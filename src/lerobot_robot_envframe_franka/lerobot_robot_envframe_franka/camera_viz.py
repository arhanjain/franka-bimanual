"""Reusable camera-grid visualization: tile named frames into one image and
stream them in a live OpenCV window.

Kept dependency-light (cv2 + numpy) and free of any robot/sim coupling so it can
be driven by anything that produces a ``{name: frame}`` dict -- ``EnvFrameFranka``
(see ``view_cameras``), a recorded dataset, or the cam-matching tools. The grid
builder is a pure function (``frames_to_grid``); ``stream_grid`` wraps it in the
window loop with quit/save keys.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def to_uint8_rgb(frame) -> np.ndarray:
    """torch/np (1,H,W,3) or (H,W,3), any dtype -> contiguous uint8 HWC RGB."""
    try:
        import torch
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
    except ImportError:
        pass
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _cell(frame, h: int, w: int, label: str | None) -> np.ndarray:
    """Resize a frame to (h, w) RGB; gray placeholder if None. Optional label."""
    import cv2

    if frame is None:
        img = np.full((h, w, 3), 64, np.uint8)
        label = (label or "") + " (no frame)"
    else:
        img = cv2.resize(to_uint8_rgb(frame), (w, h))
    if label:
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return img


def frames_to_grid(
    frames: dict[str, np.ndarray | None],
    cols: Optional[int] = None,
    tile_h: int = 240,
    labels: bool = True,
) -> np.ndarray:
    """Tile a ``{name: frame}`` dict into a single uint8 RGB grid image.

    Cells keep each frame's aspect ratio at a common ``tile_h``; widths vary per
    cell but each row is padded to a uniform width so the rows stack cleanly.
    ``cols`` defaults to ``ceil(sqrt(n))`` (roughly square). Frame order follows
    the dict's insertion order; ``None`` values render as a gray placeholder.
    """
    if not frames:
        return np.full((tile_h, tile_h, 3), 64, np.uint8)
    names = list(frames)
    n = len(names)
    cols = cols or max(1, math.ceil(math.sqrt(n)))

    cells = []
    for name in names:
        f = frames[name]
        if f is not None:
            fa = to_uint8_rgb(f)
            ar = fa.shape[1] / fa.shape[0]
        else:
            ar = 1.0
        w = max(1, round(tile_h * ar))
        cells.append(_cell(f, tile_h, w, name if labels else None))

    # Group cells into rows, then pad each row to the widest row width so hstack
    # within a row and vstack across rows both line up.
    rows = [cells[i:i + cols] for i in range(0, n, cols)]
    row_imgs = [np.hstack(r) for r in rows]
    full_w = max(im.shape[1] for im in row_imgs)
    padded = [
        np.pad(im, ((0, 0), (0, full_w - im.shape[1]), (0, 0)), constant_values=64)
        if im.shape[1] < full_w else im
        for im in row_imgs
    ]
    return np.vstack(padded)


def stream_grid(
    read_frames: Callable[[], dict[str, np.ndarray | None]],
    cols: Optional[int] = None,
    tile_h: int = 240,
    fps: float = 30.0,
    window: str = "cameras (q quit, s save)",
    save_path: str = "camera_grid.png",
) -> None:
    """Live OpenCV grid of whatever ``read_frames()`` returns each tick.

    Blocks until the user presses ``q``/ESC. ``s`` writes the current grid to
    ``save_path``. ``read_frames`` is called once per loop and may return
    ``None`` values (rendered as placeholders) without stopping the stream.
    Safe to call headless-free only -- requires a display for the window.
    """
    import cv2

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    period = 1.0 / fps if fps > 0 else 0.0
    try:
        while True:
            try:
                frames = read_frames()
            except Exception as e:  # never let a transient read kill the stream
                print(f"[camera_viz] read_frames failed: {e}")
                frames = {}
            grid = frames_to_grid(frames, cols=cols, tile_h=tile_h)
            cv2.imshow(window, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                print(f"[camera_viz] saved {save_path}")
            if period:
                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
