#!/usr/bin/env python3
"""Env-frame spacemouse teleop for the minimal EnvFrameFranka robot.

The stock ``lerobot-teleoperate`` CLI never calls ``seed_from_robot``; with an
absolute pose interface that would make the first command jump from the
spacemouse's ``initial_pos``. This launcher instead seeds each spacemouse from
the robot's measured **env-frame** EE pose before running the loop, so teleop
starts exactly where the arm already is.

Both arms are driven (two SpaceMice on separate hidraw nodes), matching the
existing scripts/spacemouse_teleop.sh setup.
"""

import logging
import signal
import threading
import time
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import tyro

from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig
from lerobot_teleoperator_spacemouse.bimanual_spacemouse import BimanualSpaceMouse
from lerobot_teleoperator_spacemouse.config_bimanual_spacemouse import BimanualSpaceMouseConfig
from lerobot_teleoperator_spacemouse.config_spacemouse import SpaceMouseConfig, SpaceMouseLeaderFields
from lerobot_teleoperator_spacemouse.spacemouse import SpaceMouse

logging.basicConfig(level=logging.INFO)
# Importing the robot package installs a root StreamHandler, making basicConfig
# above a no-op (root stays at WARNING, swallowing logger.info). Force INFO.
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("envframe_spacemouse_teleop")

_ARM_TO_LEADER = {"l": "left_arm", "r": "right_arm"}

# Built-in symmetric home pose driven before teleop starts (joint space, so both
# arms come up physically symmetric). Kept in sync with home.py's _DEFAULT_POSE:
# r_q = sim LEFT default, l_q = its exact joint mirror (j1,j3,j5,j7 negated).
_HOME_POSE = {
    "r_q": [0.6109, -0.6109, 0.0, -2.3562, 0.0, 1.8326, -0.7854],
    "l_q": [-0.6109, -0.6109, 0.0, -2.3562, 0.0, 1.8326, 0.7854],
}

# Wrist cameras are per-arm (l=left, r=right); scene cameras are always shown.
# Grid layout per row: [scene_left_0, scene_right_0, wrist_<active arms>].
_ARM_TO_WRIST = {"l": "wrist_left_plus", "r": "wrist_right_minus"}
_SCENE_VIEWS = ("scene_left_0", "scene_right_0")

# Single source of truth for env-frame SpaceMouse tuning. These OVERRIDE the
# base-frame defaults in SpaceMouseLeaderFields (which serve the base-frame
# bimanual teleop). Edit here to retune.
#   - device->env map (a +90deg-about-Z rotation): the built-in x/y swap plus
#     translation_signs; rotation reuses it (identity axis-map, env_rot=[-roll,
#     pitch,yaw]) with rotation_signs flipping direction.
#   - scales kept so full-deflection target speed (scale*fps) stays at/below the
#     arm clamp (0.30 m/s, 1.20 rad/s at 30 Hz) -> no coast after release.
_ENV_TUNING = dict(
    translation_scale=0.010,
    rotation_scale=0.040,
    translation_signs=(-1, 1, 1),
    rotation_signs=(-1, -1, -1),
    rotation_axis_map=(0, 1, 2),
)


class _CameraGrid:
    """Background thread that reads the robot's cameras and shows them tiled.

    Decoupled from the control loop: reads ``robot.cameras[*].async_read`` (which
    the control path never touches) and draws one OpenCV window. cv2 GUI calls all
    happen on this one thread. Frames are RGB from the drivers -> converted to BGR
    for cv2. View order: scene_left, scene_right, then wrist(s) for active arms.
    """

    def __init__(self, robot, view_order: list[str], fps: float = 15.0,
                 read_timeout_ms: float = 100.0, display_height: int = 480):
        self._robot = robot
        self._views = [v for v in view_order if v in robot.cameras]
        self._period = 1.0 / fps
        self._timeout = read_timeout_ms
        # Each tile is scaled to this height (px) before tiling, so the window is
        # readable -- native frames are often 224px and the grid comes out tiny.
        self._display_height = int(display_height)
        self._stop = threading.Event()
        self._quit = threading.Event()  # set when 'q' is pressed in the cv2 window
        self._thread: threading.Thread | None = None

    @property
    def quit_requested(self) -> bool:
        """True once the user pressed 'q' in the camera window (False if no grid)."""
        return self._quit.is_set()

    def start(self) -> None:
        if not self._views:
            logger.info("No cameras configured; not starting the camera grid.")
            return
        self._thread = threading.Thread(target=self._loop, name="camera-grid", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        import cv2

        win = "spacemouse teleop cameras"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            tiles = []
            for v in self._views:
                cam = self._robot.cameras[v]
                try:
                    frame = cam.async_read(self._timeout)
                except Exception:
                    frame = cam.blank_frame()
                frame = np.ascontiguousarray(frame)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Scale each tile to the display height (keep aspect ratio) so the
                # grid is readable regardless of the camera's native resolution.
                fh, fw = bgr.shape[:2]
                tw = max(1, round(fw * self._display_height / fh))
                bgr = cv2.resize(bgr, (tw, self._display_height))
                cv2.putText(bgr, v, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                tiles.append(bgr)
            # Uniform tile height already; match widths so hstack lines up.
            w = min(t.shape[1] for t in tiles)
            tiles = [cv2.resize(t, (w, self._display_height)) for t in tiles]
            cv2.imshow(win, np.hstack(tiles))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                self._quit.set()  # main loop polls quit_requested and stops
            dt = time.perf_counter() - t0
            if dt < self._period:
                self._stop.wait(self._period - dt)
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass


def _shutdown(robot, teleop) -> None:
    """Stop the arm and release devices. Safe to call even if connect() never ran."""
    # Ignore further Ctrl-C so a frantic double-press can't abort cleanup and
    # leave the arm coasting. We're exiting anyway.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Robot first: disconnect() stops all motion before closing the FCI session,
    # so the arm halts before we touch anything else.
    try:
        robot.disconnect()
    except Exception:
        logger.exception("error stopping/disconnecting robot")
    try:
        teleop.disconnect()
    except Exception:
        logger.exception("error disconnecting teleop")


@dataclass
class Args:
    """CLI args. Network defaults match EnvFrameFrankaConfig; l = mario = sim
    LEFT, r = luigi = sim RIGHT (left/right facing the robots). Mapping/scale
    tuning lives in _ENV_TUNING, not here. (Per-field code comments are avoided
    because tyro turns a comment above a field into that field's --help.)"""

    arms: Literal["l", "r", "lr"] = "lr"
    l_server_ip: str = "192.168.3.10"
    l_robot_ip: str = "192.168.201.10"
    l_port: int = 18812
    r_server_ip: str = "192.168.3.11"
    r_robot_ip: str = "192.168.200.2"
    r_port: int = 18813
    left_hidraw: str = "/dev/hidraw3"
    right_hidraw: str = "/dev/hidraw2"
    fps: float = 30.0
    twist: bool = False
    """use the legacy Cartesian-velocity twist mode instead of the default
    joint_ik (workstation DLS-IK -> streamed joint velocities)"""
    ik_hz: float = 100.0
    """inner DLS-IK loop rate (joint_ik mode)"""
    cart_gain: float = 6.0
    """joint_ik resolved-rate gain (1/s); lower if motion is springy/noisy"""
    max_joint_vel: float = 1.0
    """joint_ik per-joint velocity clamp (rad/s)"""
    # Original argparse used action="store_false" for --cameras (default True,
    # passing --cameras set it FALSE -- a latent bug). With tyro this is a plain
    # bool defaulting True, so --no-cameras disables the grid; behavior is
    # preserved because the .sh wrapper never passes --cameras.
    cameras: bool = True
    """stream a live camera grid during teleop"""
    grid_fps: float = 15.0
    """camera-grid refresh rate"""
    grid_height: int = 480
    """per-tile display height in px for the camera grid (native frames are often 224 -> too small)"""


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    arms = tuple(args.arms)  # ("l",), ("r",), or ("l", "r")

    robot_cfg = EnvFrameFrankaConfig(
        l_server_ip=args.l_server_ip, l_robot_ip=args.l_robot_ip, l_port=args.l_port,
        r_server_ip=args.r_server_ip, r_robot_ip=args.r_robot_ip, r_port=args.r_port,
        active_arms=arms,
        control_mode="twist" if args.twist else "joint_ik",
        ik_hz=args.ik_hz,
        cart_gain=args.cart_gain,
        max_joint_vel=args.max_joint_vel,
        enable_cameras=args.cameras,  # default rig (scene cams + wrist per active arm)
        enable_grippers=True,  # WSG per active arm, driven by the SpaceMouse buttons
    )

    leader = SpaceMouseLeaderFields(**_ENV_TUNING)
    hidraw = {"l": args.left_hidraw, "r": args.right_hidraw}

    bimanual = len(arms) == 2
    if bimanual:
        print(f"Using both arms...")
        teleop = BimanualSpaceMouse(BimanualSpaceMouseConfig(
            id="envframe_spacemouse_teleop",
            left_arm_config=replace(leader, hidraw_path=args.left_hidraw),
            right_arm_config=replace(leader, hidraw_path=args.right_hidraw),
        ))
    else:
        arm = arms[0]
        teleop = SpaceMouse(SpaceMouseConfig(
            id=f"envframe_spacemouse_teleop_{arm}",
            hidraw_path=hidraw[arm],
            **{f.name: getattr(leader, f.name)
               for f in leader.__dataclass_fields__.values() if f.name != "hidraw_path"},
        ))

    robot = EnvFrameFranka(robot_cfg)

    # Ctrl-C during the loop sets a flag instead of raising, so the loop exits
    # BETWEEN ticks. Interrupting mid-send_action would abandon an in-flight RPyC
    # request on a pool worker; that wedged worker then blocks process exit (the
    # ThreadPoolExecutor atexit join waits on it) and, because _shutdown has set
    # SIGINT to SIG_IGN, a second Ctrl-C can't free it -> "won't quit" (kill -9).
    # The handler is installed only once the loop owns the main thread; before
    # that (the lengthy FCI connect-retry window) the default KeyboardInterrupt
    # still aborts straight into the finally.
    stop_requested = False

    def _request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logger.info("Ctrl-C received; finishing current tick, then stopping.")

    # Grid view order: scene cams, then wrist(s) for active arms (built only when
    # --cameras populated robot.cameras).
    grid = _CameraGrid(
        robot,
        view_order=list(_SCENE_VIEWS) + [_ARM_TO_WRIST[a] for a in arms],
        fps=args.grid_fps,
        display_height=args.grid_height,
    )

    try:
        robot.connect()
        teleop.connect()
        grid.start()

        # Home the active arms (joint-space PD) before teleop so every session
        # starts from the same symmetric pose. Runs before seeding so the
        # SpaceMouse is seeded from the HOMED EE pose (no startup jump). The IK
        # control thread hasn't started yet (it starts on the first send_action),
        # so home()'s joint-velocity stream owns the arms here.
        targets_q = {a: np.asarray(_HOME_POSE[f"{a}_q"], dtype=np.float64) for a in arms}
        logger.info("Homing arms=%s before teleop...", args.arms)
        if robot.home(targets_q):
            logger.info("Homing converged.")
        else:
            logger.warning("Homing timed out before reaching tolerance; continuing.")

        # Seed from the robot's env-frame EE so there is no startup jump.
        env_poses = robot.current_ee_pose_env()
        for arm, (pos, quat_xyzw) in env_poses.items():
            leader_obj = getattr(teleop, _ARM_TO_LEADER[arm]) if bimanual else teleop
            leader_obj.seed_state(pos, quat_xyzw)
            logger.info("Seeded %s arm from env-frame EE pos=%s", arm, pos)

        signal.signal(signal.SIGINT, _request_stop)
        period = 1.0 / args.fps
        logger.info("Starting teleop loop at %.1f Hz, arms=%s (Ctrl-C or 'q' in camera window to stop).",
                    args.fps, args.arms)
        while not stop_requested:
            t0 = time.perf_counter()
            raw = teleop.get_action()
            # Bimanual teleop already emits l_/r_ prefixes; single arm does not.
            action = raw if bimanual else {f"{arms[0]}_{k}": v for k, v in raw.items()}
            # The SpaceMouse leader emits gripper in [0,1] (button-latched); the
            # robot expects the gripper field in METERS (0..0.1). Remap so the
            # buttons span the full 0->0.1 m stroke (left=close->0, right=open->0.1).
            for arm in arms:
                k = f"{arm}_gripper"
                if k in action:
                    action[k] = float(action[k]) * 0.1
            robot.send_action(action)
            if grid.quit_requested:
                logger.info("'q' pressed in camera window; stopping.")
                break
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        # Ctrl-C during connect/seed, before the loop installed its handler.
        logger.info("Ctrl-C received during startup; stopping and disconnecting.")
    finally:
        grid.stop()  # stop reading cameras before disconnect() tears them down
        _shutdown(robot, teleop)


if __name__ == "__main__":
    main()
