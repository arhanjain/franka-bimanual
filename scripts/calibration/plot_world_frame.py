"""Plot every calibrated entity (cameras + robot bases) in the env/world frame.

Reads everything from results/<view>.json: camera-in-env poses from any
`extrinsics` block (camera_calibration.py) and per-arm calibrated bases from any
`base_in_env` block (wrist_handeye.py). Arms without a calibrated base fall back
to the EnvFrameFrankaConfig.base_in_env default. Each entity is drawn as an XYZ
axis triad (R=x, G=y, B=z) at its position+orientation, so you can eyeball where
calibration thinks each camera and arm sits and which way it points.

  python plot_world_frame.py     # -> world_frame_layout.png

Quaternions in results/ are WXYZ (sim/diffik convention); base_in_env is WXYZ
too. Camera +z is the optical (look) axis -- the blue arrow points where the
camera looks.
"""

import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")  # x, y, z


# --------------------------------------------------------------------------
# 3D arrow helper (same pattern as scripts/plot_env_frame.py)
# --------------------------------------------------------------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kw):
        super().__init__((0, 0), (0, 0), *args, **kw)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


def triad(ax, origin, R, length, alpha=1.0, lw=2.5):
    """Draw an XYZ axis triad (R=x, G=y, B=z) for rotation matrix R at origin."""
    for col_idx, color in enumerate(AXIS_COLORS):
        vec = R[:, col_idx]
        ax.add_artist(Arrow3D(
            [origin[0], origin[0] + length * vec[0]],
            [origin[1], origin[1] + length * vec[1]],
            [origin[2], origin[2] + length * vec[2]],
            mutation_scale=12, arrowstyle="-|>", color=color, alpha=alpha, lw=lw,
        ))


def _R_from_quat_wxyz(q):
    qw, qx, qy, qz = q
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy wants xyzw


# --------------------------------------------------------------------------
# data sources
# --------------------------------------------------------------------------
def load_cameras():
    """[(name, position(3,), R(3,3), reproj_err_or_None)] from results/*.json."""
    cams = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        ext = data.get("extrinsics")
        if not ext or ext.get("frame") != "env":
            continue
        name = data.get("view", os.path.splitext(os.path.basename(path))[0])
        p = np.array(ext["translation_xyz"], dtype=float)
        R = _R_from_quat_wxyz(ext["quaternion_wxyz"])
        cams.append((name, p, R, ext.get("reproj_error_px")))
    return cams


def _calibrated_bases():
    """{arm: (xyz, quat_wxyz)} from any results/*.json carrying a calibrated base.

    Non-destructive preview of the wrist hand-eye result before it's committed to
    EnvFrameFrankaConfig.base_in_env. Accepts the wrist_handeye.py schema (a
    top-level `base_in_env` block with arm/translation_xyz/quaternion_wxyz) and,
    for back-compat, the older `handeye.base_in_env` nesting."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        b = data.get("base_in_env")
        if (not b or "arm" not in b) and isinstance(data.get("handeye"), dict):
            he = data["handeye"]
            if "base_in_env" in he and "arm" in he:  # legacy nesting
                b = {"arm": he["arm"], **he["base_in_env"]}
        if not b or "arm" not in b:
            continue
        out[b["arm"]] = (b["translation_xyz"], b["quaternion_wxyz"])
    return out


def load_robot_bases():
    """[(label, position(3,), R(3,3))] for each arm.

    Prefers a per-arm calibrated base from results/<view>.json (base_in_env) when
    present -- a non-destructive preview of the wrist hand-eye result -- and
    falls back to EnvFrameFrankaConfig.base_in_env (or the verbatim defaults if
    the package can't be imported, e.g. a venv without the plugin)."""
    calibrated = _calibrated_bases()
    labels = {"l": "base_l (mario)", "r": "base_r (luigi)"}
    out = []
    for key in calibrated:
        pos, quat = calibrated[key]
        label = labels.get(key, f"base_{key}") + " [calib]"
        out.append((label, np.array(pos, dtype=float), _R_from_quat_wxyz(quat)))
    return out


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
def main():
    out_path = os.path.join(RESULTS_DIR, "world_frame_layout.png")

    cams = load_cameras()
    bases = load_robot_bases()
    if not cams:
        print(f"[plot] no camera extrinsics in {RESULTS_DIR} -- "
              f"run camera_calibration.py extrinsics first. Plotting bases + env only.")

    # collect all points to auto-scale the axes
    pts = [np.zeros(3)] + [p for _, p, _, _ in cams] + [p for _, p, _ in bases]
    pts = np.array(pts)
    center = pts.mean(axis=0)
    span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    span = max(span, 0.5)
    L = 0.10 * span  # triad arm length, scaled to the scene

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # env/world frame at origin (bigger, so it reads as the reference)
    triad(ax, [0, 0, 0], np.eye(3), L * 1.6, alpha=0.9, lw=3)
    ax.scatter([0], [0], [0], color="k", s=60)
    ax.text(0, 0, L * 1.7, "env origin", color="k", fontsize=10, weight="bold")

    # robot bases (square marker)
    for name, p, R in bases:
        triad(ax, p, R, L)
        ax.scatter([p[0]], [p[1]], [p[2]], color="k", marker="s", s=55)
        ax.text(p[0], p[1], p[2] + L * 0.6, name, color="k", fontsize=9)

    # cameras (diamond marker); blue z-arrow is the look direction
    for name, p, R, err in cams:
        triad(ax, p, R, L)
        ax.scatter([p[0]], [p[1]], [p[2]], color="0.25", marker="D", s=55)
        tag = name if err is None else f"{name}\n(reproj {err:.1f}px)"
        ax.text(p[0], p[1], p[2] + L * 0.6, tag, color="0.15", fontsize=9)

    ax.set_xlabel("env x (m)")
    ax.set_ylabel("env y (m)")
    ax.set_zlabel("env z (m)")
    ax.set_title("Calibrated world layout — XYZ triads (R=x, G=y, B=z; cam +z = look axis)\n"
                 f"{len(cams)} camera(s) with extrinsics, {len(bases)} robot base(s)",
                 fontsize=11)

    # equal-ish cube around the data center
    half = span * 0.75
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=24, azim=-60)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[plot] wrote {out_path}")

    # dump the numbers too
    for name, p, R, err in cams:
        look = R[:, 2]
        print(f"cam  {name:18s} pos={p.round(3)}  look(+z)={look.round(3)}"
              + (f"  reproj={err:.2f}px" if err is not None else ""))
    for name, p, R in bases:
        print(f"base {name:18s} pos={p.round(3)}  +x={R[:,0].round(3)}")


if __name__ == "__main__":
    main()
