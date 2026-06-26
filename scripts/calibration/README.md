# Real-world calibration

Calibrates the rig's cameras and the arm base frames against a chessboard fixed
flat at the **env origin**. Two tools, split by camera type:

| Tool | Cameras | Recovers |
|---|---|---|
| [camera_calibration.py](camera_calibration.py) | FRAMOS **scene** cams (`scene_left_0`, `scene_right_0`) | intrinsics (factory read) + camera-in-env extrinsics |
| [wrist_handeye_calibration.py](wrist_handeye_calibration.py) | ARV **wrist** cams (`wrist_left_plus`, `wrist_right_minus`) | wrist intrinsics + **base-in-env** (refines `base_in_env`) + wrist-cam-in-EE mount |

[plot_world_frame.py](plot_world_frame.py) renders everything (cameras + bases)
as XYZ triads in the env frame.

Cameras are opened through the `EnvFrameFranka` rig, so frames arrive at the
exact policy resolution and processing path the policy/sim see (FRAMOS scene
640x480 native; ARV wrist oversampled->INTER_AREA to 960x600).

## Before you start

- Run from the repo root with the **rig venv** (NOT `~/.venv`, which is the sim
  venv and lacks the camera/robot plugin packages):
  ```bash
  cd ~/franka_ws/third_party/franka-bimanual   # or wherever this repo lives
  ```
  Prefix commands with `.venv/bin/python` (examples below do this).
- Chessboard: **8x13 inner corners, 20 mm squares**. If yours differs, edit
  `PATTERN` / `SQUARE_SIZE` in [calib_common.py](calib_common.py) (intrinsics
  don't depend on square size; extrinsic distances do).
- **Board placement (both tools share this convention):** lay the board flat at
  the env origin — board origin (corner `(0,0)`) at the env origin, board `+X`
  (short side, 8 corners) along env `+x` (toward the workspace), `+Y` along env
  `+y`, `+Z` up. This is what ties the recovered poses to the env frame.

## Scene cameras — `camera_calibration.py`

**Intrinsics** (factory read off the RealSense device; no board needed). Reads
both scene cams at once (pass `--view` for one):
```bash
.venv/bin/python scripts/calibration/camera_calibration.py framos-intrinsics
```
Distortion is zero (RealSense color is rectified). `FramosCamera.connect()` also
reads these live at runtime, so this command just records them into `results/`.

**Extrinsics** (camera-in-env via solvePnP; needs intrinsics first). With the
board at the env origin and arms clear of the view:
```bash
.venv/bin/python scripts/calibration/camera_calibration.py capture    --view scene_right_0 --for extrinsics
.venv/bin/python scripts/calibration/camera_calibration.py extrinsics  --view scene_right_0
```
Capture keys: `SPACE`/`c` save, `u` undo, `q`/`ESC` stop. Images go to
`extrinsics/<view>/`. The solver forces the board `+Z` up (cameras are above the
board) and averages the pose over all frames.

## Wrist cameras + base frame — `wrist_handeye_calibration.py`

Eye-in-hand robot-world/hand-eye calibration. Drives ONE arm through a precoded
sequence of env-frame poses (no teleop) over the board at the env origin,
capturing `(wrist image, raw base-frame EE pose O_T_EE)` at each settled pose,
then runs `cv2.calibrateRobotWorldHandEye` to recover **both** the wrist
intrinsics, the precise **base-in-env** transform (refining the rough hardcoded
`base_in_env`), and the wrist-cam-in-EE mount. Run one arm at a time.

The only required input is `--arm` (the tool moves that arm and reads its wrist
cam, so it won't guess); everything else has a default. `run` is the default
mode, so the whole calibration is one line:

```bash
# home, drive, capture, calibrate (board at env origin; arm moves over it)
.venv/bin/python scripts/calibration/wrist_handeye_calibration.py --arm r

# preview the trajectory + per-pose reach first (NO connect, NO motion)
.venv/bin/python scripts/calibration/wrist_handeye_calibration.py dry-run --arm r

# recompute offline from a saved capture (re-tune method, NO motion)
.venv/bin/python scripts/calibration/wrist_handeye_calibration.py from-images --arm r --method li
```

Notes:
- `run` homes the arm to a safe joint pose first (`--no-home` to skip), then at
  each waypoint resends the absolute target until the measured pose is within
  `--pos-tol-mm` / `--rot-tol-deg` (else skips the pose), and captures only when
  the board is detected. Captures + `poses.json` are saved to
  `wrist_handeye/<view>/` even on early Ctrl-C, so `from-images` can reuse them.
- The trajectory sweeps a hand-vetted reachable env-frame box (`RIGHT_BOX` in the
  script; the left arm mirrors it across the env x-axis): a 3x3x2 position grid,
  each pose aiming the wrist-cam optical axis at the **env origin** (place the
  board centered there). The wide spread of look-at directions across the box
  gives the **rotational diversity hand-eye requires** (pure translation is
  degenerate); the grid + two heights give translation/scale diversity for
  intrinsics. Edit `RIGHT_BOX` to change the swept region.
- `dry-run` flags poses beyond ~0.82 m base reach (warnings — they just fail to
  converge and are skipped at run time). Reach grows with height, so the high/far
  corners of the box may be unreachable; ~12-13 of 20 poses are reachable per arm
  with the default box, above the `--min-poses 8` floor.
- solvePnP's planar pose ambiguity can flip a near-fronto-parallel board to its
  mirror twin; the solver rejects such gross closure-residual outliers and
  re-solves on the inliers. Cross-check `--method shah` vs `li` — they should
  agree within noise.

## Plot — `plot_world_frame.py`

```bash
.venv/bin/python scripts/calibration/plot_world_frame.py --show               # Open3D viewer (mouse orbit/zoom), z=0 grid
.venv/bin/python scripts/calibration/plot_world_frame.py --show --from-results # prefer calibrated base over config default
```
Reads camera extrinsics and (with `--from-results`) the calibrated
`handeye.base_in_env` from `results/*.json`. Without `--show` it just writes
`world_frame_layout.png`. `--mpl` uses the matplotlib window instead.

## Output

`results/<view>.json` per view (blocks merged in place):
```json
{
  "view": "wrist_right_minus",
  "image_size": [960, 600],
  "intrinsics": { "source": "chessboard", "matrix": [[...]], "distortion": [...], "reproj_error_px": 0.31 },
  "extrinsics": { "frame": "env", "translation_xyz": [...], "quaternion_wxyz": [...] },
  "handeye": {
    "method": "CALIB_ROBOT_WORLD_HAND_EYE_SHAH", "arm": "r", "n_poses": 16,
    "base_in_env": { "translation_xyz": [...], "quaternion_wxyz": [...] },
    "cam_in_ee":   { "translation_xyz": [...], "quaternion_wxyz": [...] },
    "delta_vs_old": { "translation_mm": 14.2, "rotation_deg": 1.8 },
    "residuals": { "intrinsics_reproj_px": 0.31, "handeye_pos_rmse_mm": 3.4, "handeye_rot_rmse_deg": 0.6 }
  }
}
```

Quality targets: intrinsics reproj **< 0.5 px**; hand-eye pos RMSE single-digit
mm, rot RMSE **< ~1°**. A `delta_vs_old` rotation > 5° vs the current config base
prints a warning (likely board-placement or pose issue).

## Wiring results back in

Results are written non-destructively; apply them yourself after review:
- **FRAMOS intrinsics** are read live at `connect()` — no config edit needed
  (the `intrinsic_matrix` in `config_framos.py` is an identity fallback).
- **FRAMOS extrinsics** (`r_cam_in_world` / `t_cam_in_world` in
  `config_framos.py`) are stale shared defaults; `get_depth()` uses them to lift
  the point cloud into the env frame. Update per camera from the extrinsics here.
- **base_in_env** (`_DEFAULT_BASE_IN_ENV` in `envframe_franka_config.py`): replace
  the per-arm `(xyz, quat_wxyz)` with the calibrated `handeye.base_in_env` after
  previewing with `plot_world_frame.py --from-results`.

## Known issue

The bimanual stack configures the FRAMOS cams at 224x224, which is not a
supported D415e color mode — `pipeline.start` rejects it. The envframe rig used
for calibration uses 640x480 (supported), so calibration is unaffected, but the
bimanual FRAMOS path needs a supported stream size + software resize before it
will connect.
