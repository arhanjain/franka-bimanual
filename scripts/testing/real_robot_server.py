#!/usr/bin/env python3
"""Websocket bridge exposing the real bimanual EnvFrameFranka to a sim process.

This is the REAL half of the sim-vs-real drift harness. The sim rollout
(``real_vs_sim_rollout.py``) runs in the sim venv (IsaacLab, py3.11, numpy 1.x);
this server runs in the real venv (``franka-bimanual/.venv``, py3.12, numpy 2.x,
Aravis/FRAMOS). The two venvs are hard-incompatible, so they talk over a
websocket instead of sharing a process. ``openpi_client.msgpack_numpy`` frames
numpy arrays cleanly across the numpy 1.x/2.x boundary (present in both venvs).

The sim side drives; this server is a thin RPC shell over ``EnvFrameFranka``:

  request {"cmd": "home",   "l_q": [7], "r_q": [7]} -> {"ok": true}
      Joint-space home each active arm to the given joint angles (the SAME
      post-reset joint config the sim env is in), so real and sim start aligned.
  request {"cmd": "action", "sim16": [16]}          -> obs dict
      Apply one absolute sim 16-vec action (BimanualAction.from_sim_flat),
      then return the real observation: per-arm env-frame EE pose +
      the 4 policy-view camera frames (uint8 HWC RGB).
  request {"cmd": "obs"}                             -> obs dict
      Return the real observation without commanding anything.
  request {"cmd": "rest"}                            -> {"ok": true}
      Stop the joint-velocity stream (let the arm hold at rest).

obs dict (msgpack): {
    "l_pos": (3,), "l_quat_xyzw": (4,), "r_pos": (3,), "r_quat_xyzw": (4,),
    "l_gripper": float, "r_gripper": float,        # meters (when enabled)
    "scene_left_0": (H,W,3) uint8, "scene_right_0": ..., "wrist_left_plus": ...,
    "wrist_right_minus": ...,                       # only views the cameras expose
}

The action stream is sim-paced (sim steps, then sends): the joint_ik inner loop
holds the last target between actions, exactly as in live inference. Cameras and
grippers default to the same physical map as the other rig scripts -- CONFIRM the
mounting before trusting the views.

Run on franka@deepblue with the REAL venv active:
  python scripts/testing/real_robot_server.py --port 9001
"""

from __future__ import annotations

import logging
import signal
import threading
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import tyro
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("real_robot_server")

# Policy camera views the sim side expects on the real obs (subset of those the
# configured cameras actually expose; a missing one is simply absent, not fatal).
POLICY_VIEWS: tuple[str, ...] = (
    "scene_left_0", "scene_right_0", "wrist_left_plus", "wrist_right_minus",
)


@dataclass
class Args:
    port: int = 9001
    """websocket port the sim client connects to"""
    host: str = "0.0.0.0"
    arms: Literal["l", "r", "lr"] = "lr"
    twist: bool = False
    """legacy Cartesian twist mode instead of joint_ik"""
    no_grippers: bool = False
    """do not actuate the WSG grippers (pose-only)"""
    no_cameras: bool = False
    """skip cameras (pose-only drift; the grid will have no real frames)"""
    # Cameras (IPs/serials/resolution/view names) come from the
    # EnvFrameFrankaConfig default rig -- the single source of truth -- so there
    # are no per-camera flags here. Edit envframe_franka_config.py to change them.
    l_server_ip: str = "192.168.3.10"
    l_robot_ip: str = "192.168.201.10"
    l_port: int = 18812
    r_server_ip: str = "192.168.3.11"
    r_robot_ip: str = "192.168.200.2"
    r_port: int = 18813


class RealRobotService:
    """Serialises the websocket RPCs onto the single EnvFrameFranka connection."""

    def __init__(self, robot, arms: tuple[str, ...]) -> None:
        self.robot = robot
        self.arms = arms
        self._lock = threading.Lock()  # one robot; serialise concurrent clients

    def _obs(self) -> dict:
        """Real obs -> flat msgpack-friendly dict (env-frame EE pose + cam frames)."""
        obs = self.robot.get_observation()
        out: dict = {}
        for arm in self.arms:
            out[f"{arm}_pos"] = np.array(
                [obs[f"{arm}_{k}"] for k in ("x", "y", "z")], dtype=np.float64
            )
            out[f"{arm}_quat_xyzw"] = np.array(
                [obs[f"{arm}_{k}"] for k in ("qx", "qy", "qz", "qw")], dtype=np.float64
            )
            if f"{arm}_gripper" in obs:
                out[f"{arm}_gripper"] = float(obs[f"{arm}_gripper"])
        for view in POLICY_VIEWS:
            frame = obs.get(view)
            if frame is not None:
                out[view] = np.ascontiguousarray(np.asarray(frame), dtype=np.uint8)
        return out

    def handle(self, req: dict) -> dict:
        from lerobot_robot_envframe_franka import BimanualAction

        cmd = req.get("cmd")
        with self._lock:
            if cmd == "home":
                # Stop the joint_ik stream FIRST. It holds the last action target
                # at config.ik_hz forever (only disconnect() stops it), so homing
                # while it runs pits home()'s 30 Hz JV-PD loop against the 100 Hz
                # IK stream on the same arm/connection -- competing velocity
                # commands that trip "multiple motions"/reflex faults and can drop
                # the libfranka link. The next "action" restarts the thread.
                self.robot._stop_ik_thread()
                targets = {}
                for arm in self.arms:
                    q = req.get(f"{arm}_q")
                    if q is not None:
                        targets[arm] = np.asarray(q, dtype=np.float64)
                ok = self.robot.home(targets, max_time_s=req.get("max_time_s", 10.0),
                                     tol_rad=req.get("tol_rad", 0.03))
                return {"ok": bool(ok)}

            if cmd == "action":
                action = BimanualAction.from_sim_flat(np.asarray(req["sim16"], dtype=np.float64))
                print(action)
                self.robot.send_action(action)
                return self._obs()

            if cmd == "obs":
                return self._obs()

            if cmd == "rest":
                # Stop the joint_ik thread so the arm actually rests; a bare
                # stop_jv would be overwritten by the next IK tick 10 ms later.
                # _stop_ik_thread() also issues the final stop_all_joint_motion.
                try:
                    self.robot._stop_ik_thread()
                except Exception as e:
                    logger.warning("rest: _stop_ik_thread failed: %s", e)
                return {"ok": True}

            return {"error": f"unknown cmd {cmd!r}"}


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    arms = tuple(args.arms)

    from lerobot_robot_envframe_franka import EnvFrameFranka, EnvFrameFrankaConfig

    # Cameras come from the EnvFrameFrankaConfig DEFAULT RIG (the single source of
    # truth for view names + IPs + per-camera resolution): leave `cameras` empty
    # and let __post_init__ build the canonical rig (scene_*_0 + one wrist per
    # active arm). --no-cameras disables vision entirely.
    robot_cfg = EnvFrameFrankaConfig(
        l_server_ip=args.l_server_ip, l_robot_ip=args.l_robot_ip, l_port=args.l_port,
        r_server_ip=args.r_server_ip, r_robot_ip=args.r_robot_ip, r_port=args.r_port,
        active_arms=arms,
        control_mode="twist" if args.twist else "joint_ik",
        enable_cameras=not args.no_cameras,
        enable_grippers=not args.no_grippers,
    )
    robot = EnvFrameFranka(robot_cfg)

    logger.info("Connecting real robot (arms=%s, grippers=%s, cameras=%s) ...",
                args.arms, not args.no_grippers, not args.no_cameras)
    robot.connect()
    service = RealRobotService(robot, arms)

    packer = msgpack_numpy.Packer()

    def handler(conn):
        peer = conn.remote_address
        logger.info("Client connected: %s", peer)
        try:
            for raw in conn:
                req = msgpack_numpy.unpackb(raw)
                try:
                    resp = service.handle(req)
                except Exception as e:
                    logger.exception("RPC %r failed", req.get("cmd"))
                    resp = {"error": str(e)}
                conn.send(packer.pack(resp))
        except Exception as e:
            logger.info("Client %s disconnected: %s", peer, e)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    server = serve(handler, args.host, args.port, max_size=None, compression=None)
    logger.info("Real robot server listening on ws://%s:%d", args.host, args.port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stop.wait()
    finally:
        logger.info("Shutting down ...")
        server.shutdown()
        try:
            robot.robot_manager.stop_all_joint_motion()
        except Exception:
            pass
        robot.disconnect()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
