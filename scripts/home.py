"""Save / apply a named joint-space home pose for the EnvFrameFranka robot.

EE-pose homing on a 7-DOF arm can't make the two arms look symmetric: it pins
only the EE pose and leaves the redundant elbow/wrist free, so each arm settles
into a different joint config (and a near-reach-limit target also sags). Homing
in *joint* space pins all 7 DOF, so a mirror-symmetric joint target produces
physically symmetric arms. EnvFrameFranka.home() runs the joint-velocity PD.

Poses live in `~/franka_ws/home_poses/<name>.json` (same format the joint-mode
BimanualFranka used):

  {"l_q": [q1..q7], "r_q": [q1..q7]}

Subcommands:

  save NAME    Read the current joint angles and write the file. --arm l|r
               updates one arm and preserves the other's saved q. --mirror l|r
               reads only that arm and writes BOTH arms (the other = its
               left/right mirror) -- guide one arm, get a symmetric pair.
  apply NAME   Load the pose (or the built-in symmetric default) and drive the
               active arm(s) there until |q-q_target| < tol (or --max-time-s).
  list         Print the saved pose names.

Usage:
$ python scripts/home.py save  home_pose --mirror l   # guide left, mirror to right
$ python scripts/home.py apply home_pose
$ python scripts/home.py apply home_pose --arm r
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig

POSES_DIR = Path(__file__).resolve().parent.parent / "home_poses"

logger = logging.getLogger("home")

# Left/right mirror sign pattern for an FR3 mounted as a mirror of its partner:
# joints 1,3,5,7 negate, 2,4,6 stay. Applied as q_other = MIRROR_SIGN * q_ref.
MIRROR_SIGN = np.array([-1, 1, -1, 1, -1, 1, -1], dtype=np.float64)

# Built-in home, used by `apply` when no saved <name>.json exists. A symmetric
# mirror pair. Left/right are as seen facing the arms: code l = mario (sim RIGHT
# arm, env +y), code r = luigi (sim LEFT arm, env -y). r_q is the sim LEFT default
# (robot.py LEFT_PANDA_DEFAULT_JOINT_POS) and l_q is its exact joint mirror
# (j1,j3,j5,j7 negated), giving EEs symmetric across the env X-Z plane (equal x,z;
# mirrored y, ~3mm in x). The sim's own RIGHT default is NOT used because its j1
# (+0.44) isn't the mirror of left's (-0.61), leaving the EE x ~6cm asymmetric.
_DEFAULT_POSE = {
    "r_q": [0.6109, -0.6109, 0.0, -2.3562, 0.0, 1.8326, -0.7854],
    "l_q": [-0.6109, -0.6109, 0.0, -2.3562, 0.0, 1.8326, 0.7854],
}

# l = mario = sim LEFT, r = luigi = sim RIGHT (left/right as seen facing robots).
_RIG = dict(
    l_server_ip="192.168.3.10", l_robot_ip="192.168.201.10", l_port=18812,
    r_server_ip="192.168.3.11", r_robot_ip="192.168.200.2", r_port=18813,
)


def _make_robot(args: argparse.Namespace, arms: tuple[str, ...]) -> EnvFrameFranka:
    cfg = EnvFrameFrankaConfig(
        l_server_ip=args.l_server_ip, l_robot_ip=args.l_robot_ip, l_port=args.l_port,
        r_server_ip=args.r_server_ip, r_robot_ip=args.r_robot_ip, r_port=args.r_port,
        active_arms=arms,
    )
    return EnvFrameFranka(cfg)


def _path_for(name: str) -> Path:
    return POSES_DIR / f"{name}.json"


def cmd_save(args: argparse.Namespace) -> None:
    path = _path_for(args.name)
    # --mirror connects only the reference arm; otherwise connect the --arm set.
    connect_arms = (args.mirror,) if args.mirror else tuple(args.arm)
    robot = _make_robot(args, connect_arms)
    # connect() inside the try so a Ctrl-C during the connect-retry window still
    # disconnects (releasing any arm that did come up).
    try:
        robot.connect()
        kin = robot.robot_manager.current_kinematic_state_batch(list(connect_arms))
        pose = json.loads(path.read_text()) if path.exists() else {}
        if args.mirror:
            q = np.asarray(kin[args.mirror][0], dtype=np.float64)
            ref, other = args.mirror, ("r" if args.mirror == "l" else "l")
            pose[f"{ref}_q"] = [float(x) for x in q]
            pose[f"{other}_q"] = [float(x) for x in MIRROR_SIGN * q]
        else:
            for arm in connect_arms:
                pose[f"{arm}_q"] = [float(x) for x in kin[arm][0]]

        missing = [k for k in ("l_q", "r_q") if k not in pose]
        if missing:
            raise SystemExit(
                f"Cannot save: no joint values for {missing} (arm not active and no "
                f"existing {path.name} to inherit from). Save with --arm lr or --mirror first."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pose, indent=2) + "\n")
        print(json.dumps(pose, indent=2))
        print(f"\nSaved to {path}")
    finally:
        robot.disconnect()


def cmd_apply(args: argparse.Namespace) -> None:
    path = _path_for(args.name)
    if path.exists():
        pose = json.loads(path.read_text())
    else:
        logger.warning("No %s; using built-in symmetric default home.", path.name)
        pose = _DEFAULT_POSE

    arms = tuple(args.arm)
    targets_q = {arm: np.asarray(pose[f"{arm}_q"], dtype=np.float64) for arm in arms}
    robot = _make_robot(args, arms)
    # connect() inside the try so a Ctrl-C during the connect-retry window still
    # disconnects (releasing any arm that did come up).
    try:
        robot.connect()
        ok = robot.home(targets_q, max_time_s=args.max_time_s, tol_rad=args.tol_rad, fps=args.fps)
        print("apply: converged" if ok else "apply: timed out before reaching tolerance")
    finally:
        robot.disconnect()


def cmd_list(_: argparse.Namespace) -> None:
    if not POSES_DIR.exists():
        print(f"(no poses saved yet; {POSES_DIR} doesn't exist)")
        return
    names = sorted(p.stem for p in POSES_DIR.glob("*.json"))
    if not names:
        print(f"(no poses in {POSES_DIR})")
        return
    for n in names:
        print(n)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    rig = argparse.ArgumentParser(add_help=False)
    rig.add_argument("--l-server-ip", default=_RIG["l_server_ip"])
    rig.add_argument("--l-robot-ip", default=_RIG["l_robot_ip"])
    rig.add_argument("--l-port", type=int, default=_RIG["l_port"])
    rig.add_argument("--r-server-ip", default=_RIG["r_server_ip"])
    rig.add_argument("--r-robot-ip", default=_RIG["r_robot_ip"])
    rig.add_argument("--r-port", type=int, default=_RIG["r_port"])
    rig.add_argument("--arm", choices=("l", "r", "lr"), default="lr",
                     help="Which arm(s) to connect and home. Default: both.")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp_save = sub.add_parser("save", parents=[rig], help="Read and save the current joint pose")
    sp_save.add_argument("name", help="Pose name (stored as home_poses/NAME.json)")
    sp_save.add_argument("--mirror", choices=("l", "r"), default=None,
                         help="Read only this arm and write BOTH (other = its mirror).")
    sp_save.set_defaults(func=cmd_save)

    sp_apply = sub.add_parser("apply", parents=[rig], help="Drive the arms to a saved joint pose")
    sp_apply.add_argument("name", help="Pose name")
    sp_apply.add_argument("--max-time-s", type=float, default=10.0)
    sp_apply.add_argument("--fps", type=float, default=30.0)
    sp_apply.add_argument("--tol-rad", type=float, default=0.02)
    sp_apply.set_defaults(func=cmd_apply)

    sp_list = sub.add_parser("list", help="List saved pose names")
    sp_list.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
