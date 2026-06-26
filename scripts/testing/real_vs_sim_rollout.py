"""Run a policy in sim and mirror every action onto the real robot, side by side.

This is the SIM half of the sim-vs-real drift harness. It runs the normal
IsaacLab LBM rollout (infer -> step), but each step it ALSO streams the same
absolute sim 16-vec action to the real bimanual EnvFrameFranka -- which lives in
a different, incompatible venv (py3.12 / numpy 2.x / Aravis), so it runs as a
separate process behind ``real_robot_server.py`` and we talk to it over a
websocket (``openpi_client.msgpack_numpy`` frames numpy across the venv split).

Inference is ONLY on the sim env; the real robot just tracks the same actions so
we can watch how far it drifts from sim under an identical action stream. The
real arm is homed to sim's exact post-reset joint configuration each episode, so
both start aligned and the divergence we see is the open-loop sim2real gap.

Output:
  - live cv2 window: top row = sim's 4 policy-view frames, bottom row = the real
    robot's 4 camera frames (same view names), tiled per step.
  - ``<run_folder>/drift.npz``: per-step sim vs real env-frame EE pose (per arm
    pos + quat) and the commanded 16-vec, for offline drift plots.

Two venvs / two processes:
  # terminal 1 (real venv: third_party/franka-bimanual/.venv)
  python scripts/testing/real_robot_server.py --port 9001
  # terminal 2 (sim venv: sim-improvement/.venv)
  python third_party/.../scripts/testing/real_vs_sim_rollout.py \
      --environment LBM-Scenario-ImplicitIK-Vision \
      --policy.client LbmOpenpi --policy.host <policy> --policy.port 8000 \
      --run_folder runs/real_vs_sim --instruction "..." --real-port 9001
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs

# The 4 policy camera views compared between sim and real, in tile order.
POLICY_VIEWS = ("scene_left_0", "scene_right_0", "wrist_left_plus", "wrist_right_minus")


@dataclass
class RolloutArgs(EvalArgs):
    scene_path: str | None = None       # config JSON (ScenarioIKRolloutCfg)
    library_dir: str | None = None      # shared USD model library
    rollouts: int = 10
    overwrite: bool = False
    real_host: str = "127.0.0.1"        # real_robot_server host
    real_port: int = 9001
    arms: str = "lr"                    # arms present on BOTH sim and real
    no_window: bool = False             # disable the live cv2 grid window
    hold: bool = False                  # HOLD mode: no policy; command the sim
    #                                     reset pose as a constant so both sim and
    #                                     real just sit at home (static drift +
    #                                     camera alignment, no motion).


class RealLink:
    """Blocking RPC client to ``real_robot_server`` (msgpack-numpy over websocket)."""

    def __init__(self, host: str, port: int):
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        self._packer = msgpack_numpy.Packer()
        self._unpack = msgpack_numpy.unpackb
        uri = f"ws://{host}:{port}"
        print(f"[real_vs_sim] connecting to real robot server at {uri} ...")
        self._conn = websockets.sync.client.connect(uri, max_size=None, compression=None)
        print("[real_vs_sim] connected.")

    def rpc(self, req: dict) -> dict:
        self._conn.send(self._packer.pack(req))
        resp = self._unpack(self._conn.recv())
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"real server error for cmd={req.get('cmd')!r}: {resp['error']}")
        return resp

    def action(self, sim16: np.ndarray) -> dict:
        return self.rpc({"cmd": "action", "sim16": np.asarray(sim16, dtype=np.float64)})

    def rest(self) -> dict:
        return self.rpc({"cmd": "rest"})

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def _sim_ee(obs, arm) -> tuple[np.ndarray, np.ndarray]:
    """Sim env-frame EE pose (pos, quat_wxyz) of env 0 for arm key l/r."""
    side = "left" if arm == "l" else "right"
    v = obs["vision"]
    pos = v[f"{side}_ee_pos"][0].detach().cpu().numpy().astype(np.float64)
    quat_wxyz = v[f"{side}_ee_quat"][0].detach().cpu().numpy().astype(np.float64)
    return pos, quat_wxyz


def _hold_action16(obs) -> np.ndarray:
    """Constant sim 16-vec that commands the CURRENT sim EE pose for both arms.

    Layout matches the policy output / BimanualAction.from_sim_flat:
      [L xyz(3), L quat_wxyz(4), R xyz(3), R quat_wxyz(4), L grip, R grip].
    Gripper fields are 0 (held). Used by --hold to keep both arms at the reset
    pose so sim and real just sit at home for static comparison."""
    lp, lq = _sim_ee(obs, "l")
    rp, rq = _sim_ee(obs, "r")
    return np.concatenate([lp, lq, rp, rq, [0.0, 0.0]]).astype(np.float64)


def _to_uint8_hwc(frame) -> np.ndarray:
    """Torch/np (H,W,3) -> contiguous uint8 numpy HWC."""
    try:
        import torch
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
    except ImportError:
        pass
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _tile_row(frames: list, h: int = 224) -> np.ndarray:
    """hstack frames at common height h; gray placeholder for any missing one."""
    import cv2

    row = []
    for f in frames:
        if f is None:
            row.append(np.full((h, h, 3), 64, np.uint8))
            continue
        if f.shape[0] != h:
            w = max(1, round(f.shape[1] * h / f.shape[0]))
            f = cv2.resize(f, (w, h))
        row.append(f)
    return np.hstack(row)


def _drift(sim_pos, sim_quat_wxyz, real_pos, real_quat_xyzw) -> tuple[float, float]:
    """(position L2 error [m], orientation geodesic angle [rad]) sim vs real."""
    from scipy.spatial.transform import Rotation

    pos_err = float(np.linalg.norm(np.asarray(sim_pos) - np.asarray(real_pos)))
    w, x, y, z = sim_quat_wxyz
    r_sim = Rotation.from_quat([x, y, z, w])
    r_real = Rotation.from_quat(real_quat_xyzw)
    ang_err = float((r_sim.inv() * r_real).magnitude())
    return pos_err, ang_err


def main(eval_args: RolloutArgs):
    # >>>> Isaac Sim App Launcher (must precede any IsaacLab import) <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import sim_improvement.environments  # noqa: F401  (registers LBM-Scenario-* ids)
    from sim_improvement.inference.lbm_openpi_client import LbmOpenpiClient
    from sim_improvement.inference.zero_action_client import ZeroActionClient
    from sim_improvement.inference.droid_jointpos import DroidJointPosClient

    CLIENT_REGISTRY = {
        "DroidJointPos": DroidJointPosClient,
        "LbmOpenpi": LbmOpenpiClient,
        "Zero": ZeroActionClient,
    }

    arms = tuple(eval_args.arms)
    if not eval_args.no_window:
        import cv2  # noqa: F401  (fail early if the GUI build is missing)

    env_cfg = parse_env_cfg(eval_args.environment, device="cuda", num_envs=1, use_fabric=True)
    env_cfg.dynamic_setup(scene_path=eval_args.scene_path, library_dir=eval_args.library_dir)  # type: ignore

    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=eval_args.overwrite)

    env = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

    client_cls = CLIENT_REGISTRY.get(eval_args.policy.client)
    if client_cls is None:
        raise ValueError(f"Unknown client {eval_args.policy.client!r}. "
                         f"Available: {list(CLIENT_REGISTRY)}")
    if env.action_space.shape[0] != 1:
        raise ValueError("real_vs_sim compares ONE real robot; run with num_envs=1.")

    # HOLD mode needs no policy server -- the action is the frozen sim reset pose.
    policy_client = None
    if not eval_args.hold:
        policy_client = client_cls(
            host=eval_args.policy.host, port=eval_args.policy.port,
            open_loop_horizon=eval_args.policy.open_loop_horizon,
            action_shape=env.action_space.shape,
        )
    instruction = eval_args.instruction or "put the red bell pepper in the bin"

    real = RealLink(eval_args.real_host, eval_args.real_port)
    drift_log: list[dict] = []
    drift_path = run_folder / "drift.npz"

    def save_drift():
        """Write the whole drift_log to drift.npz (atomic via a .tmp rename).

        Called incrementally (per episode + every SAVE_EVERY steps), NOT only at
        the end: IsaacLab/Omniverse installs its own SIGINT handler, so Ctrl-C
        hard-exits the process and the finally block below never runs. Frequent
        saves mean a Ctrl-C still leaves a complete log on disk."""
        if not drift_log:
            return
        payload = {
            "steps": np.array([d["step"] for d in drift_log]),
            "arms": np.array([d["arm"] for d in drift_log]),
            "sim_pos": np.stack([d["sim_pos"] for d in drift_log]),
            "sim_quat_wxyz": np.stack([d["sim_quat_wxyz"] for d in drift_log]),
            "real_pos": np.stack([d["real_pos"] for d in drift_log]),
            "real_quat_xyzw": np.stack([d["real_quat_xyzw"] for d in drift_log]),
            "sim16": np.stack([d["sim16"] for d in drift_log]),
        }
        # Tmp name MUST end in .npz: np.savez auto-appends ".npz" otherwise, so
        # a ".npz.tmp" name would be written as ".npz.tmp.npz" and the replace
        # below would miss it.
        tmp = drift_path.with_name("drift.tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(drift_path)

    SAVE_EVERY = 25  # steps

    # Belt-and-suspenders against Omniverse's hard-exit SIGINT handler: flush the
    # drift log on Ctrl-C before re-raising so the finally / KeyboardInterrupt
    # path can run too. (Registered after AppLauncher so it wins over carb's.)
    import signal

    def _on_sigint(*_args):
        save_drift()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    def show_grid(sim_obs, real_obs, step, errs):
        if eval_args.no_window:
            return
        import cv2

        sim_row = _tile_row([_to_uint8_hwc(sim_obs["vision"][v][0]) for v in POLICY_VIEWS])
        real_row = _tile_row([
            _to_uint8_hwc(real_obs[v]) if v in real_obs else None for v in POLICY_VIEWS
        ])
        # Match row dims (height too) so the rows AND the per-pixel overlay align.
        if sim_row.shape[:2] != real_row.shape[:2]:
            h = min(sim_row.shape[0], real_row.shape[0])
            w = min(sim_row.shape[1], real_row.shape[1])
            sim_row = cv2.resize(sim_row, (w, h))
            real_row = cv2.resize(real_row, (w, h))
        # Overlay row: sim and real blended 50/50, so a mismatch shows as ghosting.
        overlay_row = cv2.addWeighted(sim_row, 0.5, real_row, 0.5, 0.0)

        grid = np.vstack([sim_row, real_row, overlay_row])

        label = "  ".join(f"{a}:dp={errs[a][0]*1000:.0f}mm da={np.degrees(errs[a][1]):.0f}deg"
                          for a in arms)
        cv2.putText(grid, f"SIM / REAL / OVERLAY  step {step}  {label}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("real_vs_sim", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    obs, _ = env.reset()
    if policy_client is not None:
        policy_client.reset()

    # HOLD mode: freeze the action at the sim reset pose so both sim and real
    # just sit at home -- no policy server needed, no motion. Captured once.
    hold_sim16 = _hold_action16(obs) if eval_args.hold else None
    if eval_args.hold:
        print("[real_vs_sim] HOLD mode: commanding the sim reset pose as a "
              "constant (no policy). Ctrl-C to stop.")

    successful_episodes = 0
    step = 0
    try:
        # NOTE: no real-arm homing here -- the real robot just tracks whatever
        # action stream the sim policy produces from its current pose. Make sure
        # the real arms are already at a sane starting pose before launching.
        while True:
            if eval_args.hold:
                sim16 = hold_sim16  # constant home pose; no inference
            else:
                action, _ = policy_client.infer(obs, instruction=instruction, return_viz=False)
                sim16 = np.asarray(action[0], dtype=np.float64)  # absolute 16-vec, env 0

            # Mirror the action onto the real robot and read its resulting obs.
            try:
                real_obs = real.action(sim16)
            except Exception as e:
                print(f"[real_vs_sim] real action failed: {e}")
                real_obs = {}

            # Step sim with the SAME action (hold: re-commands the home pose).
            action_t = torch.tensor(sim16[None], dtype=torch.float32) if eval_args.hold \
                else torch.tensor(action, dtype=torch.float32)
            obs, _, term, trunc, _ = env.step(action_t)

            # Drift: sim env-frame EE pose vs real env-frame EE pose, per arm.
            errs = {}
            for arm in arms:
                sim_pos, sim_quat = _sim_ee(obs, arm)
                rp, rq = real_obs.get(f"{arm}_pos"), real_obs.get(f"{arm}_quat_xyzw")
                if rp is not None and rq is not None:
                    errs[arm] = _drift(sim_pos, sim_quat, rp, rq)
                else:
                    errs[arm] = (float("nan"), float("nan"))
                drift_log.append({
                    "step": step, "arm": arm, "sim_pos": sim_pos, "sim_quat_wxyz": sim_quat,
                    "real_pos": np.asarray(rp) if rp is not None else np.full(3, np.nan),
                    "real_quat_xyzw": np.asarray(rq) if rq is not None else np.full(4, np.nan),
                    "sim16": sim16,
                })

            if real_obs:
                show_grid(obs, real_obs, step, errs)
            step += 1
            if step % SAVE_EVERY == 0:
                save_drift()  # checkpoint mid-episode (Ctrl-C may hard-exit)

            # HOLD mode never resets/ends on termination -- it just keeps holding.
            if not eval_args.hold and (term.any() or trunc.any()):
                needs_reset = (term | trunc).nonzero().flatten().detach().cpu().numpy()
                policy_client.reset(env_ids=needs_reset, obs=obs)
                success_values = env.termination_manager.get_term("success")[needs_reset]
                successful_episodes += int(success_values.sum().item())
                save_drift()  # checkpoint at every episode boundary
                print(f"[real_vs_sim] successful episodes: {successful_episodes} "
                      f"({len(drift_log)} drift rows -> {drift_path})")
                if successful_episodes >= eval_args.rollouts:
                    break
    except KeyboardInterrupt:
        print("[real_vs_sim] stopped by user (Ctrl-C).")
    finally:
        # Persist whatever isn't already checkpointed. NOTE: IsaacLab's SIGINT
        # handler can hard-exit the process before this runs, which is why
        # save_drift() is also called inline above -- this is the best-effort
        # final flush, not the only save.
        save_drift()
        if drift_log:
            print(f"[real_vs_sim] drift log -> {drift_path} ({len(drift_log)} rows)")
        try:
            real.rest()
        except Exception:
            pass
        real.close()
        if not eval_args.no_window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    args: RolloutArgs = tyro.cli(RolloutArgs)
    main(args)
