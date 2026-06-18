"""Jog ONE arm a small step along ONE env-frame axis, to verify the env mapping.

Joint homing can't confirm the env (EE-frame) left/right assignment because it
never uses base_in_env, and a mirror of a symmetric reset still looks symmetric.
This exercises the env path directly: seed the current env pose, add `delta` on
one world axis, and track it for a few seconds. Watch which PHYSICAL arm moves
and in which direction:

  - `--arm l --axis y --delta 0.08`  -> the arm code calls "l" should move, and
    it should move toward env +Y. If the wrong physical arm moves, the NUC<->arm
    (or base_in_env) assignment is swapped. If the right arm moves the wrong way,
    base_in_env orientation for that arm is off.

Env axes (world-aligned): +X, +Y, +Z(up). In the sim scene left_panda sits at
-Y, right_panda at +Y.

Usage:
$ python scripts/misc/env_jog.py --arm l --axis z --delta 0.05
"""

import argparse
import time

import numpy as np

from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("l", "r"), required=True)
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--delta", type=float, default=0.05, help="step in metres (default 0.05)")
    p.add_argument("--time", type=float, default=3.0, help="seconds to track the target")
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    robot = EnvFrameFranka(EnvFrameFrankaConfig(active_arms=(args.arm,)))
    robot.connect()
    try:
        pos, quat = robot.current_ee_pose_env()[args.arm]
        target_pos = np.array(pos, dtype=np.float64)
        target_pos["xyz".index(args.axis)] += args.delta
        print(f"{args.arm}: env {args.axis} {pos['xyz'.index(args.axis)]:+.3f} -> {target_pos['xyz'.index(args.axis)]:+.3f}")

        action = {f"{args.arm}_{k}": float(v) for k, v in zip("xyz", target_pos)}
        action.update({f"{args.arm}_{k}": float(v) for k, v in zip(("qx", "qy", "qz", "qw"), quat)})

        period = 1.0 / args.fps
        end = time.perf_counter() + args.time
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            robot.send_action(action)
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
        robot.robot_manager.stop_all_motion()

        final = robot.current_ee_pose_env()[args.arm][0]
        print(f"{args.arm}: env pose now {np.round(final, 3)}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
