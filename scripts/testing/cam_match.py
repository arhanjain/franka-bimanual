#!/usr/bin/env python3
"""Live camera-alignment view: sim reference (frozen) vs real (live), per view.

For physically aiming the real cameras to match the sim views. The sim frames
are a FROZEN snapshot loaded from a pickle (IsaacLab can't run in the real venv);
the real frames stream live from EnvFrameFranka's cameras. Three rows, columns =
the shared camera views (scene_left_0, scene_right_0, wrist_left_plus,
wrist_right_minus):

    row 0  SIM   (frozen reference)
    row 1  REAL  (live)
    row 2  OVERLAY (sim & real blended 50/50)

Move/aim each real camera until its column's overlay lines up with the sim row.

Real frames come through the normal ``EnvFrameFranka.connect()`` /
``get_observation()`` path (same as the harness), so the chosen ``--arms`` must
be online -- use ``--arms r`` if only luigi is up.

Run in the REAL venv (third_party/franka-bimanual/.venv), e.g.:
  python scripts/testing/cam_match.py --pkl /home/qirico/qirico/sim-improvement/img_data.pkl
  python scripts/testing/cam_match.py --arms r   # only luigi (right) online

Keys in the window: q/ESC quit, [ / ] decrease/increase overlay opacity,
s save the current grid to <out>.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tyro

# Camera views compared, in column order. These are the keys EnvFrameFranka's
# observation exposes (cameras are built with these names below) and the keys in
# the sim pkl's payload["sim"]["vision"] / payload["real"].
POLICY_VIEWS = ("scene_left_0", "scene_right_0", "wrist_left_plus", "wrist_right_minus")


@dataclass
class Args:
    pkl: str = "/home/qirico/qirico/sim-improvement/img_data.pkl"
    """pickle with payload['sim']['vision'][view] (torch/np HWC) for the frozen sim row"""
    arms: Literal["l", "r", "lr"] = "lr"
    out: str = "cam_match.png"
    """path written when you press 's'"""
    tile_h: int = 240
    """per-cell display height in px (width scales to keep aspect)"""
    alpha: float = 0.5
    """initial overlay opacity of the sim frame over the real frame"""


def _to_uint8_rgb(frame) -> np.ndarray:
    """torch/np (1,H,W,3) or (H,W,3) -> contiguous uint8 HWC RGB."""
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


def _load_sim(pkl_path: str) -> dict:
    """{view: uint8 HWC RGB} frozen sim reference for the views we have."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    vision = payload["sim"]["vision"]
    out = {}
    for view in POLICY_VIEWS:
        if view in vision:
            out[view] = _to_uint8_rgb(vision[view])
    return out


def _cell(frame: np.ndarray | None, h: int, w: int, label: str) -> np.ndarray:
    """Resize a frame to (h, w) RGB; gray placeholder if None. Adds a label."""
    if frame is None:
        img = np.full((h, w, 3), 64, np.uint8)
    else:
        img = cv2.resize(frame, (w, h))
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    arms = tuple(args.arms)

    sim = _load_sim(args.pkl)
    views = [v for v in POLICY_VIEWS if v in sim] or list(POLICY_VIEWS)
    print(f"[cam_match] sim views: {list(sim.keys())}")

    from lerobot_robot_envframe_franka import EnvFrameFrankaConfig, EnvFrameFranka

    # Default camera rig (both scene cams + the wrist cam per active arm), keyed
    # with the canonical sim/policy view names -- no custom build needed.
    cfg = EnvFrameFrankaConfig(
        active_arms=arms,
        enable_grippers=False,
        enable_cameras=True,
    )
    robot = EnvFrameFranka(cfg)

    # Full EnvFrameFranka.connect(): brings up the arms (RPyC) AND the cameras,
    # so reads go through robot.get_observation() like the real harness. The arms
    # for the chosen --arms must be up; use --arms r if only luigi is online.
    print(f"[cam_match] connecting EnvFrameFranka (arms={args.arms}) + cameras ...")
    robot.connect()

    h = args.tile_h
    alpha = float(args.alpha)
    win = "cam_match (q quit, [ ] opacity, s save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[cam_match] streaming; aim the real cameras to match the sim row.")

    try:
        while True:
            # Read live real frames via the EnvFrameFranka observation (cameras
            # are keyed by view name in the obs dict, same as the real harness).
            try:
                obs = robot.get_observation()
            except Exception as e:
                print(f"[cam_match] get_observation failed: {e}")
                obs = {}
            real = {}
            for view in views:
                frame = obs.get(view)
                real[view] = _to_uint8_rgb(frame) if frame is not None else None

            # Uniform cell width per column from the sim aspect (fallback square).
            cols = []
            for view in views:
                s = sim.get(view)
                r = real.get(view)
                ar = (s.shape[1] / s.shape[0]) if s is not None else (
                    (r.shape[1] / r.shape[0]) if r is not None else 1.0)
                w = max(1, round(h * ar))
                s_cell = _cell(s, h, w, f"SIM {view}")
                r_cell = _cell(r, h, w, f"REAL {view}")
                # Overlay: blend sim over real at the same cell size.
                if s is not None and r is not None:
                    base_s = cv2.resize(s, (w, h))
                    base_r = cv2.resize(r, (w, h))
                    ov = cv2.addWeighted(base_s, alpha, base_r, 1.0 - alpha, 0.0)
                elif s is not None:
                    ov = cv2.resize(s, (w, h))
                elif r is not None:
                    ov = cv2.resize(r, (w, h))
                else:
                    ov = np.full((h, w, 3), 64, np.uint8)
                cv2.putText(ov, f"OVL a={alpha:.2f}", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cols.append(np.vstack([s_cell, r_cell, ov]))

            grid = np.hstack(cols)
            cv2.imshow(win, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("["):
                alpha = max(0.0, alpha - 0.05)
            elif key == ord("]"):
                alpha = min(1.0, alpha + 0.05)
            elif key == ord("s"):
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                print(f"[cam_match] saved {args.out}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[cam_match] done.")


if __name__ == "__main__":
    main()
