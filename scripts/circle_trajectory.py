#!/usr/bin/env python3
"""Play an open-loop circular EE trajectory on the real EnvFrameFranka and log it.

For sim/real alignment: drives one or both arms around a circle in a chosen
env-frame plane (default the z-y plane) using the same absolute-pose interface
as teleop, while recording the commanded targets, the measured EE poses, and
the measured joint angles ``q`` at every tick. The resulting ``.npz`` is the
ground-truth real trajectory; replay the logged env-frame targets through the
sim ``LBM-Scenario-ImplicitIK-State`` action and diff ``sim_q`` against the
logged ``meas_q`` to get per-joint tracking error.

The circle is constructed to pass through each arm's *current* EE pose at
phase 0 (center = current - radius * u_axis), so there is no startup jump.
Orientation is held fixed at the measured start quaternion per arm.
"""

import argparse
import logging
import signal
import time

import numpy as np

from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("envframe_circle_trajectory")

EE_AXIS_KEYS = ("x", "y", "z", "qx", "qy", "qz", "qw")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _plane_axes(plane: str) -> tuple[int, int]:
    """'zy' -> (u=z-index, v=y-index). First char sweeps via cos, second via sin."""
    plane = plane.lower()
    if len(plane) != 2 or any(c not in _AXIS_INDEX for c in plane) or plane[0] == plane[1]:
        raise ValueError(f"--plane must be two distinct axes from x/y/z, e.g. 'zy'; got {plane!r}")
    return _AXIS_INDEX[plane[0]], _AXIS_INDEX[plane[1]]


def _shutdown(robot: EnvFrameFranka) -> None:
    """Stop the arm and release the session. Safe even if connect() never ran."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # finish cleanup even on double Ctrl-C
    try:
        robot.disconnect()
    except Exception:
        logger.exception("error stopping/disconnecting robot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", choices=("l", "r", "lr"), default="lr")
    # Network (defaults match EnvFrameFrankaConfig / spacemouse_teleop.sh).
    p.add_argument("--l-server-ip", default="192.168.3.11")
    p.add_argument("--l-robot-ip", default="192.168.200.2")
    p.add_argument("--l-port", type=int, default=18813)
    p.add_argument("--r-server-ip", default="192.168.3.10")
    p.add_argument("--r-robot-ip", default="192.168.201.10")
    p.add_argument("--r-port", type=int, default=18812)
    # Trajectory shape.
    p.add_argument("--plane", default="zy", help="two env-frame axes for the circle, e.g. 'zy' (default), 'xy'")
    p.add_argument("--radius", type=float, default=0.08, help="circle radius in meters")
    p.add_argument("--period", type=float, default=8.0, help="seconds per revolution")
    p.add_argument("--revolutions", type=float, default=2.0, help="number of revolutions to trace")
    p.add_argument("--ramp", type=float, default=2.0,
                   help="seconds to ease radius 0->full at start and full->0 at end (no velocity jolt)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out", default=None, help="output .npz path (default: circle_traj_<timestamp>.npz)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arms = tuple(args.arms)  # ("l",), ("r",), or ("l", "r")
    u_axis, v_axis = _plane_axes(args.plane)

    robot_cfg = EnvFrameFrankaConfig(
        l_server_ip=args.l_server_ip, l_robot_ip=args.l_robot_ip, l_port=args.l_port,
        r_server_ip=args.r_server_ip, r_robot_ip=args.r_robot_ip, r_port=args.r_port,
        active_arms=arms,
    )
    robot = EnvFrameFranka(robot_cfg)

    period = 1.0 / args.fps
    duration = args.period * args.revolutions
    n_steps = int(round(duration * args.fps))

    # Per-arm logs: commanded target + measured EE + measured joints, per tick.
    log: dict[str, dict[str, list]] = {
        arm: {"target_pos": [], "target_quat": [], "meas_pos": [], "meas_quat": [], "meas_q": []}
        for arm in arms
    }
    timestamps: list[float] = []

    # connect() inside try so a Ctrl-C during the FCI connect-retry window still
    # runs _shutdown and releases the session.
    try:
        robot.connect()

        # Seed circle anchor + fixed orientation from the measured env-frame EE.
        start = robot.current_ee_pose_env()  # {arm: (xyz, quat_xyzw)}
        starts: dict[str, np.ndarray] = {}
        quats: dict[str, np.ndarray] = {}
        for arm, (pos, quat_xyzw) in start.items():
            # Anchor the circle on the START pose, not a separate center: the
            # offset below is zero at phase 0 AND at full revolutions, so the
            # ramp returns exactly to start (not to the center one radius away).
            starts[arm] = np.asarray(pos, dtype=np.float64).copy()
            quats[arm] = np.asarray(quat_xyzw, dtype=np.float64)
            logger.info("Seeded %s arm: start=%s (plane=%s, r=%.3f)", arm, starts[arm], args.plane, args.radius)

        omega = 2.0 * np.pi / args.period
        logger.info(
            "Tracing %.1f rev(s) over %.1fs at %.1f Hz on arms=%s (Ctrl-C to stop).",
            args.revolutions, duration, args.fps, args.arms,
        )

        t_start = time.perf_counter()
        for step in range(n_steps):
            t0 = time.perf_counter()
            t = step / args.fps

            # Ease the amplitude in/out about the start pose so the EE speed
            # ramps from rest. The offset is anchored at start (cos(0)-1 = 0),
            # so scale->0 returns to start rather than drifting to a center.
            scale = 1.0
            if args.ramp > 0.0:
                scale = min(scale, t / args.ramp, max(0.0, (duration - t) / args.ramp))
            theta = omega * t
            du = args.radius * scale * (np.cos(theta) - 1.0)
            dv = args.radius * scale * np.sin(theta)

            action: dict[str, float] = {}
            for arm in arms:
                target = starts[arm].copy()
                target[u_axis] += du
                target[v_axis] += dv
                q_xyzw = quats[arm]
                for key, val in zip(EE_AXIS_KEYS, (*target, *q_xyzw)):
                    action[f"{arm}_{key}"] = float(val)

            robot.send_action(action)

            # Record what we commanded and what the robot actually did this tick.
            kin = robot.robot_manager.current_kinematic_state_batch(list(arms))
            meas = robot.current_ee_pose_env()
            timestamps.append(t0 - t_start)
            for arm in arms:
                q = kin[arm][0]  # snapshot is (q, dq, O_T_EE, twist)
                m_pos, m_quat = meas[arm]
                log[arm]["target_pos"].append([action[f"{arm}_{k}"] for k in ("x", "y", "z")])
                log[arm]["target_quat"].append([action[f"{arm}_{k}"] for k in ("qx", "qy", "qz", "qw")])
                log[arm]["meas_pos"].append(np.asarray(m_pos, dtype=np.float64))
                log[arm]["meas_quat"].append(np.asarray(m_quat, dtype=np.float64))
                log[arm]["meas_q"].append(np.asarray(q, dtype=np.float64))

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        logger.info("Ctrl-C received; stopping the arm and disconnecting.")
    finally:
        _shutdown(robot)

    # Persist whatever was collected (even on early Ctrl-C) for sim replay/diff.
    if timestamps:
        out_path = args.out or f"circle_traj_{int(time.time())}.npz"
        payload: dict[str, np.ndarray] = {
            "arms": np.array(arms),
            "timestamps": np.asarray(timestamps),
            "fps": np.asarray(args.fps),
            "plane": np.array(args.plane),
            "radius": np.asarray(args.radius),
            "period": np.asarray(args.period),
            "joint_names": np.array([f"panda_joint{i}" for i in range(1, 8)]),
            "quat_order": np.array("xyzw"),
        }
        for arm in arms:
            for field, rows in log[arm].items():
                payload[f"{arm}_{field}"] = np.asarray(rows, dtype=np.float64)
        np.savez(out_path, **payload)
        logger.info("Saved %d ticks to %s", len(timestamps), out_path)


if __name__ == "__main__":
    main()
