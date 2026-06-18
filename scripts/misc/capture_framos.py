#!/usr/bin/env python3
"""Manually capture RGB frames from both FRAMOS D415e cameras.

Standalone — does not touch the FR3 control stack or LeRobot CLI. Talks
straight to the cameras via `lerobot_camera_framos.FramosCamera`.

Press Enter to capture one frame from each camera; `q` + Enter (or Ctrl+C)
to quit. Frames are written to OUT as PNGs named `{ts}_{shot:04d}_{slot}.png`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from lerobot_camera_framos import FramosCamera, FramosCameraConfig


# IPs/serials mirror BimanualFrankaConfig.
CAMERAS: tuple[tuple[str, str, str, str], ...] = (
    ("cam_2", "workspace_framos_d71", "192.168.0.116", "6CD146030D71"),
    ("cam_6", "workspace_framos_d63", "192.168.1.102", "6CD146030D63"),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "franka_data" / "demo_images",
        help="Output directory for captured PNGs.",
    )
    p.add_argument("--width", type=int, default=1280, help="Color frame width.")
    p.add_argument("--height", type=int, default=720, help="Color frame height.")
    p.add_argument("--fps", type=int, default=30, help="Streaming FPS (6/15/30/60/90).")
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Frames to discard per camera so auto-exposure settles.",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cams: list[tuple[str, FramosCamera]] = []
    try:
        for slot, name, ip, sn in CAMERAS:
            cfg = FramosCameraConfig(
                name=name,
                ip=ip,
                serial_number=sn,
                fps=args.fps,
                width=args.width,
                height=args.height,
                color_width=args.width,
                color_height=args.height,
                enable_color=True,
                enable_depth=False,
            )
            cam = FramosCamera(cfg)
            cam.connect()
            cams.append((slot, cam))

        for slot, cam in cams:
            for _ in range(args.warmup_frames):
                try:
                    cam.async_read()
                except Exception:
                    pass

        print(f"Ready. Saving to {args.out}.")
        print("Press Enter to capture; type 'q' + Enter (or Ctrl+C) to quit.")
        shot = 0
        while True:
            try:
                line = input("> ")
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if line.strip().lower() in {"q", "quit", "exit"}:
                break
            ts = time.strftime("%Y%m%d_%H%M%S")
            for slot, cam in cams:
                frame = cam.read()  # RGB
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                path = args.out / f"{ts}_{shot:04d}_{slot}.png"
                cv2.imwrite(str(path), bgr)
                print(f"  saved {path}")
            shot += 1
    finally:
        for _, cam in cams:
            try:
                cam.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
