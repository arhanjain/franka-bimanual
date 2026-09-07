# Real-world calibration

Calibrates the rig's cameras and the arm base frames against a **ChArUco board
laid flat with its CENTER at the env origin**. Two tools, split by camera type:

| Tool | Cameras | Recovers |
|---|---|---|
| [camera_calibration.py](camera_calibration.py) | FRAMOS **scene** cams (`scene_left_0`, `scene_right_0`) | intrinsics (factory read) + camera-in-env extrinsics |
| [wrist_handeye.py](wrist_handeye.py) | ARV **wrist** cams (`wrist_left_plus`, `wrist_right_minus`) | wrist intrinsics + **base-in-env** + wrist-cam-in-EE mount |

[plot_world_frame.py](plot_world_frame.py) renders everything (cameras + bases)
as XYZ triads in the env frame.

Both tools share [calib_common.py](calib_common.py) — board geometry, ChArUco
detection, the env re-aim, and the `results/` + `samples/` JSON/image stores —
so the two paths can't drift into different world frames.

Cameras are opened through the `EnvFrameFranka` rig, so frames arrive at the
exact policy resolution and processing path the policy/sim see (FRAMOS scene
640x480 native; ARV wrist oversampled x2 → INTER_AREA to 960x600).

## Before you start

- Use the **rig venv**, NOT `~/.venv` (that's the sim venv and lacks the
  camera/robot plugin packages):
  ```bash
  cd ~/franka_ws/third_party/franka-bimanual   # or wherever this repo lives
  ```
  Prefix commands with `.venv/bin/python` (examples below do this). The scripts
  import `calib_common` by bare name, which works because Python puts the
  script's own directory on `sys.path`; `results/`/`samples/` paths are absolute,
  so the working directory doesn't matter.
- **Board: 11x11 ChArUco, `DICT_4X4_1000`, 50 mm checker squares, 39 mm markers.**
  If yours differs, edit `CHARUCO_*` in [calib_common.py](calib_common.py).
  Square size sets the metric scale; marker size only affects ArUco detection.

### Board placement + the env frame (both tools share this)

Lay the board flat so its **center** sits at the env origin, with the board's
**+X toward the workspace (env +x)**, **+Y to env +y (left)**, **+Z up**. This
placement *is* the definition of the env frame for both tools.

Two details that make this work, both in `calib_common.charuco_match_env`:

- OpenCV's ChArUco object points are anchored at a board **corner**; they get
  shifted by `-_CHARUCO_CENTER` so the frame origin is the board **center**.
- OpenCV's native board axes come out (in physical terms) +X right / +Y back /
  +Z down. `BOARD_REAIM` rotates them to the env convention +X fwd / +Y left /
  +Z up. It has `det = +1` — a **proper rotation, not a reflection**; a
  single-axis negate would be a reflection and would crash
  `Rotation.from_matrix`.

Because the object points already carry the env axes, `solvePnP` /
`calibrateCamera` return `cam_T_env` directly — no per-detection `+Z`-up flip and
no separate center shift or post-multiply downstream.

## Scene cameras — `camera_calibration.py`

One run does everything for one camera: read factory intrinsics off the
RealSense device, auto-capture board frames, then solve and save the
camera-in-env extrinsics. Only that camera is connected — **the arms are never
brought up**, so this works with the NUCs offline.

```bash
.venv/bin/python scripts/calibration/camera_calibration.py --side r   # scene_right_0
.venv/bin/python scripts/calibration/camera_calibration.py --side l   # scene_left_0
```

- `--side` follows the same convention as `wrist_handeye.py --arm`
  (`r` = RIGHT = luigi side, `l` = LEFT = mario side).
- Capture is **automatic**: the cam and the board are both fixed, so it just
  grabs `N_CAPTURES = 5` board-detected frames `CAPTURE_INTERVAL_S = 1.0` s
  apart. The live stream is shown with detected corners overlaid; `q`/`ESC` bails
  out early. Clean (un-overlaid) frames land in `samples/<view>/`, and each run
  clears that directory first so stale frames never mix into the solve.
- Intrinsics come from the device (`source: "factory"`). Distortion is zero —
  RealSense color is already rectified. `FramosCamera.connect()` reads these live
  at runtime anyway, so this step just records them into `results/`.
- Extrinsics average the per-image pose (quaternions sign-aligned to the first,
  then renormalized) and report mean reprojection error.

## Wrist cameras + base frame — `wrist_handeye.py`

Eye-in-hand robot-world/hand-eye calibration. Drives **one** arm through a
precoded env-frame trajectory over the board (no teleop), capturing
`(wrist image, achieved EE pose)` whenever the board is detected, then runs
`cv2.calibrateRobotWorldHandEye` to recover the wrist intrinsics, the
**base-in-env** transform, and the wrist-cam-in-EE mount. Run one arm at a time.

`--arm` is the only flag, and it's required (the tool moves that arm and reads
its wrist cam, so it won't guess):

```bash
.venv/bin/python scripts/calibration/wrist_handeye.py --arm r
.venv/bin/python scripts/calibration/wrist_handeye.py --arm l
```

**What the run does:**

1. Builds the default rig for that arm alone, keeps **only** its wrist-cam
   config (verbatim IP/fps/resolution) and drops the scene cams.
2. Connects (`joint_ik`) and homes to `HOME_Q_BY_ARM[arm]` — the mirror-pair safe
   pose from `scripts/home.py`.
3. Commands the first circle's start pose and streams **without saving** for
   `SETTLE_S = 3.0` s, so the home→start transit doesn't get captured as motion
   blur.
4. Traces one ramped circle in the env xy-plane at each z in
   `Z_LEVELS = (0.55, 0.50)`: `RADIUS = 0.1` m around `CENTER_XY = (-0.1, 0.1)`,
   `PERIOD = 15` s/rev, 1 revolution, at `FPS = 10`. `RAMP = 2.0` s eases the
   radius 0→full→0 so each level starts and ends exactly at the center (no
   velocity jolt entering/leaving a level).
5. At every point the EE **+Z axis (the wrist-cam optical axis)** is aimed at
   `AIM_POINT = (-0.1, -0.1, 0.0)` via `look_at_quat`. The wide spread of look-at
   directions around each circle is what gives hand-eye the **rotational
   diversity it requires** (pure translation is degenerate); the two z levels add
   translation/scale diversity for the intrinsics.
6. Captures the clean frame whenever the board is detected, throttled to
   `CAPTURE_INTERVAL_S = 0.5` s so a slow sweep doesn't dump dozens of
   near-identical views. A typical run yields ~55 usable views per arm.

The **left arm mirrors the right trajectory across the env x-axis** (`y → -y` on
both `AIM_POINT` and `CENTER_XY`); `_select_arm` applies that per `--arm`.

**Notes:**

- The saved pose is the **achieved** EE pose read from the *same*
  `get_observation()` call as the frame — not the commanded pose. Each record
  also stores the true base-frame `O_T_EE`, recovered via `robot._env_to_base`
  (the exact inverse of the transform that produced the observation, so the
  `base_in_env` config cancels out and the hand-eye anchor is independent of
  whatever base guess is currently in the config).
- Ctrl-C is handled: the arm stops, `poses.json` is written, and the solve still
  runs on whatever was captured (needs ≥ 3 detections).
- Solve orientation: with `A = cam_T_board` and `B = gripper_T_base`,
  `calibrateRobotWorldHandEye`'s `X` output is the **base pose in env**
  (== `base_in_env`) *directly*, not `base_T_env`. This is verified against
  synthetic ground truth and this dataset — it's the easiest thing to get
  backwards in this file.
- Method is fixed to `CALIB_ROBOT_WORLD_HAND_EYE_SHAH`.
- Every used frame gets a `<idx>_axes.png` sibling with the calibrated **env
  (board-center) frame** drawn on it. These come from the *full hand-eye chain*
  (`cam_T_env = inv(cam_in_ee) @ gripper_T_base @ inv(env_T_base)`), **not** a
  per-view PnP — so they show the calibration's own belief about where the world
  frame sits. **This is the primary sanity check:** if the calibration is good,
  the projected origin lands on the board center and the axes lie along the board
  edges in *every* view.

## Plot — `plot_world_frame.py`

```bash
.venv/bin/python scripts/calibration/plot_world_frame.py
```

Takes no flags. Writes `results/world_frame_layout.png` (matplotlib) and prints
each entity's position and axis directions. Reads camera-in-env poses from every
`extrinsics` block and per-arm bases from every `base_in_env` block in
`results/*.json` — so it's a non-destructive preview of a fresh hand-eye result
before you commit it to the config. Camera **+z is the optical axis**, so the
blue arrow points where the camera looks.

Note: `load_robot_bases()`'s docstring claims it falls back to
`EnvFrameFrankaConfig.base_in_env` for arms without a calibrated base; the
implementation only plots bases found in `results/`. An arm you haven't
calibrated simply won't appear.

## Output

Captured frames live in `samples/<view>/`:

- scene: `1.png` … `5.png`
- wrist: `001.png` …, plus `poses.json` (per-frame achieved env pose + `O_T_EE`)
  and the `<idx>_axes.png` verification overlays

Results are one JSON per view, `results/<view>.json`, with blocks merged in place
so the intrinsics/extrinsics/handeye steps don't clobber each other. Quaternions
are **WXYZ** (sim/diffik convention) even though scipy uses XYZW internally.

Scene cam:

```json
{
  "view": "scene_left_0",
  "image_size": [640, 480],
  "intrinsics": { "source": "factory", "matrix": [[...]], "distortion": [0,0,0,0,0],
                  "model": "distortion.inverse_brown_conrady" },
  "extrinsics": { "frame": "env", "translation_xyz": [...], "quaternion_wxyz": [...],
                  "rotation_matrix": [[...]], "n_images": 5, "reproj_error_px": 0.41 }
}
```

Wrist cam:

```json
{
  "view": "wrist_right_minus",
  "image_size": [960, 600],
  "intrinsics": { "source": "charuco_calibrateCamera", "matrix": [[...]],
                  "distortion": [...], "reproj_rms_px": 0.129, "n_views": 56 },
  "handeye": {
    "method": "shah", "n_views": 56,
    "base_T_env": [[...]],                       // env(board-center) pose in the base
    "base_to_env_translation_xyz": [...], "base_to_env_quaternion_wxyz": [...],
    "cam_in_ee_matrix": [[...]],                 // wrist cam mount on the EE
    "cam_in_ee_translation_xyz": [...], "cam_in_ee_quaternion_wxyz": [...]
  },
  "board_center_in_base_xyz": [...],             // == env origin in the base frame
  "base_in_env": { "arm": "r", "translation_xyz": [...], "quaternion_wxyz": [...] }
}
```

The top-level `base_in_env` block is shaped for direct paste-in to
`EnvFrameFrankaConfig.base_in_env[arm] = (xyz, quat_wxyz)`.

**Quality targets:** reprojection RMS **< 0.5 px** (the current wrist solves come
in near 0.13 px, scene ~0.41 px). Beyond the numbers, check the
`<idx>_axes.png` overlays — a solve with a good reprojection error can still be
mirrored or mis-anchored, and the overlays catch that where the RMS won't.

## Wiring results back in

Results are written non-destructively; apply them yourself after review.

- **`base_in_env`** — `_DEFAULT_BASE_IN_ENV` in `envframe_franka_config.py`
  **already carries the calibrated values from these runs** for both arms; the
  original sim-derived transforms are commented out just above them. Re-paste
  only if you re-calibrate, and preview with `plot_world_frame.py` first.
- **FRAMOS intrinsics** — read live at `connect()`, so no config edit is needed
  (the `intrinsic_matrix` in `config_framos.py` is only a fallback if the read
  fails).
- **FRAMOS extrinsics** — `r_cam_in_world` / `t_cam_in_world` in
  `config_framos.py` are **stale shared defaults** (one pose for both cameras).
  `get_depth()` uses them to lift the point cloud into the env frame, so update
  them per camera from the `extrinsics` block here.
- **Wrist `cam_in_ee`** — recovered but not yet consumed anywhere in the stack;
  it's there for whatever needs to project wrist observations into the env frame.

## Known issue

The **bimanual** stack (`bimanual_franka_config.py`, `config_single_arm_franka.py`)
still configures the FRAMOS cams at 224x224, which is not a supported D415e color
mode — `pipeline.start` rejects it. Calibration is unaffected, since the envframe
rig it uses runs the scene cams at 640x480; but the bimanual FRAMOS path needs a
supported stream size plus a software resize before it will connect. (The ARV
wrist cams are fine at 224 — the driver now auto-picks the largest downscale
factor ≤ 8 that fits the sensor.)
