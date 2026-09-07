#!/usr/bin/env python3
"""Live grid of the real EnvFrameFranka cameras (no sim, no arms required).

Pops up an OpenCV window tiling every connected camera (both scene cams + the
wrist cam per active arm), keyed by view name. Thin wrapper over
``EnvFrameFranka.view_cameras`` / the ``camera_viz`` util in the package, so the
same grid is reusable from any script or notebook.

By default only the CAMERAS are brought up (``connect_cameras``), so this works
even when the FR3 arms are offline. Pass ``--with-arms`` to do a full
``connect()`` and read frames through the normal observation path instead.

Run in the REAL venv (third_party/franka-bimanual/.venv):
  python scripts/view_cameras.py
  python scripts/view_cameras.py --arms r          # only luigi's wrist cam + scene cams
  python scripts/view_cameras.py --tile-h 320 --cols 2

Keys in the window: q/ESC quit, s save the current grid to <out>.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import tyro


@dataclass
class Args:
    arms: Literal["l", "r", "lr"] = "lr"
    """which arms' wrist cameras to include (scene cams are always included)"""
    with_arms: bool = False
    """also connect the FR3 arms and read via get_observation (default: cameras only)"""
    tile_h: int = 480
    """per-cell display height in px (width scales to keep each frame's aspect)"""
    cols: Optional[int] = None
    """grid columns (default: ~square, ceil(sqrt(n)))"""
    fps: float = 30.0
    """display refresh rate"""
    out: str = "camera_grid.png"
    """path written when you press 's'"""


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    arms = tuple(args.arms)

    from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig

    cfg = EnvFrameFrankaConfig(active_arms=arms, enable_grippers=False, enable_cameras=True)
    robot = EnvFrameFranka(cfg)

    if args.with_arms:
        print(f"[view_cameras] connecting arms + cameras (arms={args.arms}) ...")
        robot.connect()
        read = robot.read_camera_frames
    else:
        print("[view_cameras] connecting cameras only (arms offline OK) ...")
        robot.connect_cameras()
        read = robot.read_camera_frames

    if not robot.cameras:
        print("[view_cameras] no cameras configured; nothing to show.")
        return

    print(f"[view_cameras] streaming {list(robot.cameras)} (q/ESC quit, s save).")
    try:
        from lerobot_robot_envframe_franka import stream_grid
        stream_grid(read, cols=args.cols, tile_h=args.tile_h, fps=args.fps, save_path=args.out)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[view_cameras] done.")


if __name__ == "__main__":
    main()
