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

import argparse
import logging
import signal
import time
from dataclasses import replace

from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig
from lerobot_teleoperator_spacemouse.bimanual_spacemouse import BimanualSpaceMouse
from lerobot_teleoperator_spacemouse.config_bimanual_spacemouse import BimanualSpaceMouseConfig
from lerobot_teleoperator_spacemouse.config_spacemouse import SpaceMouseConfig, SpaceMouseLeaderFields
from lerobot_teleoperator_spacemouse.spacemouse import SpaceMouse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("envframe_spacemouse_teleop")

_ARM_TO_LEADER = {"l": "left_arm", "r": "right_arm"}

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Which arms to drive. "l" or "r" runs a single arm with one SpaceMouse;
    # "lr" runs both. Only the selected arms are connected/commanded.
    p.add_argument("--arms", choices=("l", "r", "lr"), default="lr")
    # Network (defaults match EnvFrameFrankaConfig). l = mario = sim LEFT,
    # r = luigi = sim RIGHT (left/right as seen facing the robots).
    p.add_argument("--l-server-ip", default="192.168.3.10")
    p.add_argument("--l-robot-ip", default="192.168.201.10")
    p.add_argument("--l-port", type=int, default=18812)
    p.add_argument("--r-server-ip", default="192.168.3.11")
    p.add_argument("--r-robot-ip", default="192.168.200.2")
    p.add_argument("--r-port", type=int, default=18813)
    # SpaceMouse devices.
    p.add_argument("--left-hidraw", default="/dev/hidraw3")
    p.add_argument("--right-hidraw", default="/dev/hidraw2")
    # Mapping/scale tuning lives in _ENV_TUNING (single source), not here.
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arms = tuple(args.arms)  # ("l",), ("r",), or ("l", "r")

    robot_cfg = EnvFrameFrankaConfig(
        l_server_ip=args.l_server_ip, l_robot_ip=args.l_robot_ip, l_port=args.l_port,
        r_server_ip=args.r_server_ip, r_robot_ip=args.r_robot_ip, r_port=args.r_port,
        active_arms=arms,
    )

    leader = SpaceMouseLeaderFields(**_ENV_TUNING)
    hidraw = {"l": args.left_hidraw, "r": args.right_hidraw}

    bimanual = len(arms) == 2
    if bimanual:
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

    try:
        robot.connect()
        teleop.connect()

        # Seed from the robot's env-frame EE so there is no startup jump.
        env_poses = robot.current_ee_pose_env()
        for arm, (pos, quat_xyzw) in env_poses.items():
            leader_obj = getattr(teleop, _ARM_TO_LEADER[arm]) if bimanual else teleop
            leader_obj.seed_state(pos, quat_xyzw)
            logger.info("Seeded %s arm from env-frame EE pos=%s", arm, pos)

        signal.signal(signal.SIGINT, _request_stop)
        period = 1.0 / args.fps
        logger.info("Starting teleop loop at %.1f Hz, arms=%s (Ctrl-C to stop).", args.fps, args.arms)
        while not stop_requested:
            t0 = time.perf_counter()
            raw = teleop.get_action()
            # Bimanual teleop already emits l_/r_ prefixes; single arm does not.
            action = raw if bimanual else {f"{arms[0]}_{k}": v for k, v in raw.items()}
            robot.send_action(action)
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        # Ctrl-C during connect/seed, before the loop installed its handler.
        logger.info("Ctrl-C received during startup; stopping and disconnecting.")
    finally:
        _shutdown(robot, teleop)


if __name__ == "__main__":
    main()
