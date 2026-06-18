"""Remove depth point-cloud columns from a recorded LeRobot dataset.

Depth scalars (depth_0 … depth_N) live inside observation.state alongside
joint angles.  This script copies the dataset, slices those columns out of
every parquet file, and updates the metadata accordingly.  Stats are
recomputed at the end so the new dataset is ready for training.

Usage
-----
    python scripts/strip_depth.py <repo_id> <new_repo_id> \\
        --root ~/franka_data/<src_dir> \\
        --new-root ~/franka_data/<dst_dir> \\
        [--push-to-hub]

The original dataset is never modified.  If the source is absent from the
local cache it is downloaded from HuggingFace Hub automatically.  If
--new-root is omitted the output goes to $HF_LEROBOT_HOME/<new_repo_id>.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from lerobot.utils.constants import HF_LEROBOT_HOME


def _find_depth_indices(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if n.startswith("depth_")]


def strip_depth(
    repo_id: str,
    new_repo_id: str,
    root: Path,
    new_root: Path,
    push_to_hub: bool,
) -> None:
    src = root / repo_id
    dst = new_root / new_repo_id

    if not src.is_dir():
        print(f"{src} not found locally — downloading from Hub…")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=src)
        print("Download complete.")
    if dst.exists():
        sys.exit(f"Destination already exists: {dst}  (delete it first or choose a different name)")

    # Read feature metadata from info.json
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    state_feat = info["features"].get("observation.state")
    if state_feat is None:
        sys.exit("No 'observation.state' feature found in dataset — nothing to strip.")

    all_names: list[str] = list(state_feat["names"])
    depth_indices = _find_depth_indices(all_names)
    if not depth_indices:
        sys.exit("No depth_ columns found in observation.state — nothing to strip.")

    keep_indices = [i for i in range(len(all_names)) if i not in set(depth_indices)]
    keep_names = [all_names[i] for i in keep_indices]
    print(f"Removing {len(depth_indices)} depth columns, keeping {len(keep_indices)} state features.")

    # Copy everything except data/
    print("Copying metadata and videos…")
    shutil.copytree(src / "meta", dst / "meta")
    # Keep the source stats.json — video/image stats are unchanged.
    # recompute_stats (skip_image_video=True) will update only observation.state
    # and leave the image/video entries from this file intact.

    videos_src = src / "videos"
    if videos_src.exists():
        shutil.copytree(videos_src, dst / "videos")

    # Update info.json with new state shape/names
    info["features"]["observation.state"]["shape"] = [len(keep_indices)]
    info["features"]["observation.state"]["names"] = keep_names
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # Process parquet files
    parquet_files = sorted((src / "data").rglob("*.parquet"))
    print(f"Processing {len(parquet_files)} parquet files…")
    for src_file in parquet_files:
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_file)
        if "observation.state" in df.columns:
            sliced = [[row[i] for i in keep_indices] for row in df["observation.state"].tolist()]
            df["observation.state"] = sliced
        df.to_parquet(dst_file, index=False)

    print(f"Dataset written to {dst}")

    # Recompute stats — pass the full dataset path as --root so lerobot finds
    # meta/info.json locally without falling back to a Hub lookup.
    print("Recomputing statistics…")
    edit_script = Path(__file__).parent.parent.parent / "lerobot" / "src" / "lerobot" / "scripts" / "lerobot_edit_dataset.py"
    cmd = [
        sys.executable, str(edit_script),
        "--repo_id", new_repo_id,
        "--root", str(dst),
        "--operation.type", "recompute_stats",
    ]
    if push_to_hub:
        cmd += ["--push_to_hub", "True"]
    subprocess.run(cmd, check=True)

    print("Done." if not push_to_hub else f"Done — pushed as {new_repo_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="Source dataset repo id (e.g. HuskyMango/my-dataset)")
    parser.add_argument("new_repo_id", help="Output dataset repo id (e.g. HuskyMango/my-dataset-no-depth)")
    parser.add_argument("--root", type=Path, default=HF_LEROBOT_HOME, help="Parent directory containing <repo_id>/ (default: $HF_LEROBOT_HOME)")
    parser.add_argument("--new-root", type=Path, help="Parent directory for output (default: same as --root)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the stripped dataset to HuggingFace Hub")
    args = parser.parse_args()

    new_root = args.new_root or args.root
    strip_depth(args.repo_id, args.new_repo_id, args.root, new_root, args.push_to_hub)


if __name__ == "__main__":
    main()
