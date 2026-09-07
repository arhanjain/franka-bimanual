"""Transparent data-collection wrapper for ``EnvFrameFranka``.

Real-hardware analog of the sim ``RecorderManager`` term stack
(``environments/mdp/recorders``): it wraps a live robot so the control loop calls
``get_observation`` / ``send_action`` exactly as before -- the collector is
invisible to the loop ("returns it like it was never there") -- and records each
transition to an HDF5 file in the SAME layout the sim teleop writes
(``runs/<task>/teleop.hdf5``).

Usage (teleop / replay loop)::

    robot = EnvFrameDataCollector(EnvFrameFranka(cfg), output_dir, "teleop")
    robot.connect()
    robot.start_episode()
    while running:
        obs = robot.get_observation()          # recorded as pre-step obs
        action = leader.get_action(obs)         # teleop / policy / replay
        robot.send_action(action)               # recorded as the step action
    robot.end_episode(success=True)             # flush one demo_N to disk
    robot.close()

Episodes are delimited MANUALLY: the teleop script calls ``end_episode(success)``
(typically on a keypress) and ``discard_episode()`` to drop a buffered episode
without writing. There is no env/reset/success-term machinery on real hardware.

``end_episode`` does NOT block on the (slow gzip) HDF5 write: it hands the
episode buffers to a background writer thread and returns immediately, so the
caller can reset/re-home while the previous demo writes. ALL h5py I/O lives on
that one thread (h5py is not thread-safe). ``num_recorded_demos`` advances at
hand-off (so the HUD is instant); ``pending_writes`` reports how many demos are
still in flight; ``close()`` drains the queue and joins the writer. The queue is
bounded (maxsize 2) so if writing falls behind capture, ``end_episode`` blocks
rather than letting RAM grow unbounded.

Layout per ``data/demo_{n}`` (matches the sim file; mug/object poses are omitted
because the real rig has no object-pose sensing, and ``*_panda_joint_pos`` is
7-dim here vs sim's 12 because franky exposes only the 7 arm joints)::

    action/robot__action__poses__{side}::panda__xyz            (N, 3)
    action/robot__action__poses__{side}::panda__quat_wxyz      (N, 4)
    action/robot__action__poses__{side}::panda__xyz_relative   (N, 3)
    action/robot__action__poses__{side}::panda__axis_angle_relative (N, 3)
    action/robot__action__grippers__{side}::panda_hand         (N, 1)
    obs/policy            (N, D)   concat of obs/policy_dict in sorted-key order
    obs/policy_dict/{left,right}_{ee_pos,ee_quat,gripper_pos,panda_joint_pos}
    obs/vision_dict/<view>         (N, H, W, 3) uint8  [if record_images]
    next_obs/policy_dict/...       (same low-dim keys; post-step)
    next_obs/policy                (post-step)
    states/articulation/{side}_panda/{joint_position,joint_velocity,root_pose,root_velocity}
    next_states/...       (same)

Camera frames are stored at NATIVE resolution under ``obs/vision_dict/<view>``
(view = the EnvFrameFranka camera key, e.g. ``scene_left_0``, ``wrist_left_plus``;
these match the sim ``VisionCfg`` group and the policy client's ``POLICY_VIEWS``).
Images are written for ``obs`` ONLY -- NOT ``next_obs`` -- because
``next_obs[t] == obs[t+1]``; duplicating frames would ~double the (large) file
for no new information. Reconstruct a next-image as ``obs/vision_dict[t+1]`` if a
consumer needs it. Set ``record_images=False`` for low-dim-only (sim-parity) files.

``side`` is ``left``/``right`` (sim convention; robot arm ``l``->left, ``r``->right).
Absolute action poses and EE-obs poses are the panda_link8 frame in the env frame
(same as ``EnvFrameFranka``), quaternions WXYZ. Relative action fields use the
sim convention ``desired - pre_step_current`` via the shared ``compute_pose_error``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading

import numpy as np

from .actions import BimanualAction
from .diffik import compute_pose_error
from .envframe_franka import _ARM_TO_SIDE

logger = logging.getLogger("envframe.collector")


class EnvFrameDataCollector:
    """Wrap an ``EnvFrameFranka`` and record transitions to a sim-format HDF5.

    Forwards every attribute except ``get_observation`` / ``send_action`` (which
    are intercepted to record) and the episode-control methods, so it is a drop-in
    stand-in for the robot in any control loop.
    """

    def __init__(
        self,
        robot,
        output_dir: str,
        filename: str = "teleop",
        env_name: str = "",
        record_states: bool = True,
        record_images: bool = True,
        overwrite: bool = False,
    ):
        # Set self.robot FIRST so __getattr__ (which forwards to it) never recurses
        # while the rest of __init__ runs.
        self.robot = robot
        self.output_dir = str(output_dir)
        self.filename = filename if filename.endswith(".hdf5") else filename + ".hdf5"
        self.env_name = env_name
        self.record_states = record_states
        # When True, store each camera view's RGB frame per step under
        # obs/vision_dict/<view> at NATIVE resolution (uint8 (N,H,W,3)). next_obs
        # gets NO images (next_obs[t] == obs[t+1]; reconstruct from obs if needed)
        # to avoid doubling the (large) image storage. Frames buffer in RAM until
        # end_episode -- native res is ~5 MB/step (~1 GB per 200-step demo), kept
        # bounded by the teleop reset-gate (recording only between b and s/f).
        self.record_images = record_images

        # overwrite=True REPLACES any existing file: delete it up front so the
        # default append path starts from a clean file (count seeds to 0). Default
        # False appends, never clobbering prior demos.
        if overwrite:
            path = os.path.join(self.output_dir, self.filename)
            if os.path.exists(path):
                os.remove(path)
                logger.warning("overwrite=True: removed existing %s", path)

        self._h5 = None            # lazily opened on first end_episode write
        # Seed the demo count from any existing file NOW (not lazily at first
        # write) so num_recorded_demos -- and the teleop HUD -- shows the true
        # running total from the start. Without this it reads 0 until the first
        # save, then jumps to the on-disk count (the "0 -> 8" surprise).
        self._demo_count = self._count_existing_demos()

        self._recording = False
        self._obs_buf: list[dict] = []     # per-step low-dim obs (policy_dict)
        self._img_buf: list[dict] = []     # per-step camera frames (vision_dict)
        self._state_buf: list[dict] = []   # per-step articulation state
        self._act_buf: list[dict] = []     # per-step action datasets (abs + relative)

        # Background writer: end_episode hands the buffers off to this queue and
        # returns immediately, so the (slow gzip) HDF5 write of the previous demo
        # overlaps the next reset/teleop instead of stalling it. ALL h5py I/O
        # happens on this one thread (h5py is not thread-safe). _demo_count is
        # reserved on the main thread at hand-off (so the HUD updates at once and
        # indices stay unique); the writer only creates the reserved group.
        # maxsize=2 bounds RAM if writing falls behind capture (puts block).
        self._write_q: queue.Queue = queue.Queue(maxsize=2)
        self._writer_err: Exception | None = None
        self._writer = threading.Thread(target=self._writer_loop, name="hdf5-writer", daemon=True)
        self._writer.start()

    def _count_existing_demos(self) -> int:
        """Count demo_* groups already on disk (0 if no file yet). Read-only."""
        path = os.path.join(self.output_dir, self.filename)
        if not os.path.exists(path):
            return 0
        try:
            import h5py

            with h5py.File(path, "r") as f:
                if "data" not in f:
                    return 0
                return len([k for k in f["data"].keys() if k.startswith("demo_")])
        except Exception as e:
            logger.warning("Could not count existing demos in %s: %s", path, e)
            return 0

    # ------------------------------------------------------------------
    # transparent proxy
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        # Only reached for attributes not found on the collector itself; forward
        # them to the wrapped robot so the loop sees an EnvFrameFranka.
        return getattr(self.__dict__["robot"], name)

    def get_observation(self, include_cameras: bool = True):
        obs = self.robot.get_observation(include_cameras=include_cameras)
        if self._recording:
            self._capture_obs(obs)
        return obs

    def send_action(self, action):
        dispatched = self.robot.send_action(action)
        if self._recording:
            self._capture_action(action)
        return dispatched

    # ------------------------------------------------------------------
    # episode control (called manually by the teleop/replay script)
    # ------------------------------------------------------------------
    def start_episode(self) -> None:
        """Begin buffering a new episode (discards any unfinished buffer)."""
        self._obs_buf = []
        self._img_buf = []
        self._state_buf = []
        self._act_buf = []
        self._recording = True

    def discard_episode(self) -> None:
        """Drop the buffered episode without writing it."""
        self._obs_buf = []
        self._img_buf = []
        self._state_buf = []
        self._act_buf = []
        self._recording = False

    def end_episode(self, success: bool) -> bool:
        """Finish the buffered episode and hand it to the background writer.

        Returns immediately (does NOT block on the HDF5 write) so the caller can
        reset/re-home while the previous demo's slow gzip write runs on the writer
        thread. Returns True if a demo was queued, False if there was nothing to
        write. Recording stops; call ``start_episode`` for the next.
        """
        self._recording = False
        if self._writer_err is not None:  # surface a prior writer failure
            raise RuntimeError(f"HDF5 writer thread failed: {self._writer_err}")
        n_actions = len(self._act_buf)
        if n_actions == 0:
            logger.warning("end_episode: no actions buffered; nothing written")
            self.discard_episode()
            return False

        # Each transition t needs next_obs[t] = obs[t+1], so the buffer must hold
        # one more observation than actions. Top up with a final read (on THIS
        # thread -- it touches the robot) if the loop ended right after send_action.
        if len(self._obs_buf) <= n_actions:
            try:
                self._capture_obs(self.robot.get_observation())
            except Exception as e:  # disconnected / read error: drop last action
                logger.warning("end_episode: trailing obs read failed (%s); "
                               "dropping last action", e)
        n = min(n_actions, len(self._obs_buf) - 1)
        if n <= 0:
            logger.warning("end_episode: insufficient observations; nothing written")
            self.discard_episode()
            return False

        # Reserve a demo index NOW (main thread) so the HUD count is correct
        # immediately and successive queued demos never collide on the index
        # (_demo_count was seeded from disk; advance it monotonically). Hand the
        # buffers OFF by reference (then start_episode allocates fresh lists), so
        # the writer owns this snapshot with no further mutation from the loop.
        demo_idx = self._demo_count
        self._demo_count = demo_idx + 1  # HUD reflects the queued demo at once
        self._write_q.put((demo_idx, bool(success), n,
                           self._obs_buf, self._img_buf, self._state_buf, self._act_buf))
        self.discard_episode()           # fresh buffers for the next episode
        return True

    @property
    def num_recorded_demos(self) -> int:
        """Demos on disk + queued/in-flight (the HUD count). Advances at hand-off."""
        return self._demo_count

    @property
    def pending_writes(self) -> int:
        """How many finished demos are still queued/being written (0 == flushed)."""
        return self._write_q.qsize()

    def _writer_loop(self) -> None:
        """Background thread: build + write each queued demo. Owns ALL h5py I/O."""
        while True:
            item = self._write_q.get()
            try:
                if item is None:  # sentinel: drain done, exit
                    return
                demo_idx, success, n, obs_buf, img_buf, state_buf, act_buf = item
                try:
                    episode = self._build_episode(n, obs_buf, img_buf, state_buf, act_buf)
                    self._write_demo(episode, demo_idx, success=success, num_samples=n)
                except Exception as e:
                    # Record the failure so the main thread can surface it; keep the
                    # thread alive to drain the rest rather than wedging close().
                    logger.exception("hdf5-writer: failed to write demo_%d", demo_idx)
                    self._writer_err = e
            finally:
                self._write_q.task_done()

    def close(self) -> None:
        """Flush all queued writes, stop the writer thread, close the file."""
        if self._writer is not None and self._writer.is_alive():
            self._write_q.put(None)       # sentinel after the last real item
            self._writer.join()
            self._writer = None
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None
        if self._writer_err is not None:
            raise RuntimeError(f"HDF5 writer thread failed: {self._writer_err}")

    # ------------------------------------------------------------------
    # capture helpers
    # ------------------------------------------------------------------
    def _capture_obs(self, obs: dict) -> None:
        """Buffer one step: low-dim obs terms, camera frames, articulation state.

        Splits the obs dict into camera views (keys present in robot.cameras) and
        low-dim terms. Low-dim goes to policy_dict; frames (when record_images and
        present -- the end_episode trailing read passes include_cameras=False, so
        no frame then) go to vision_dict as uint8 (H,W,3). The two buffers stay
        index-aligned with _act_buf: obs[t] pairs with action[t].
        """
        cams = getattr(self.robot, "cameras", {})
        low: dict = {}
        imgs: dict = {}
        for k, v in obs.items():
            if k in cams:
                if self.record_images and v is not None:
                    imgs[k] = np.ascontiguousarray(v, dtype=np.uint8)
            else:
                low[k] = np.asarray(v, dtype=np.float32).reshape(-1).copy()
        self._obs_buf.append(low)
        self._img_buf.append(imgs)
        if self.record_states:
            self._state_buf.append(self._capture_state())

    def _capture_state(self) -> dict:
        """Per-arm articulation state from the robot's last kinematic read."""
        kin = getattr(self.robot, "_last_kin", None)
        state: dict[str, dict] = {}
        if not kin:
            return state
        base_in_env = self.robot.config.base_in_env
        for arm in self.robot.active_arms:
            side = _ARM_TO_SIDE[arm]
            q, dq, _T_eb, _twist = kin[arm]
            (px, py, pz), (qw, qx, qy, qz) = base_in_env[arm]
            state[f"{side}_panda"] = {
                "joint_position": np.asarray(q, dtype=np.float32).reshape(-1),
                "joint_velocity": np.asarray(dq, dtype=np.float32).reshape(-1),
                "root_pose": np.array([px, py, pz, qw, qx, qy, qz], dtype=np.float32),
                "root_velocity": np.zeros(6, dtype=np.float32),  # base is static
            }
        return state

    def _capture_action(self, action) -> None:
        """Record absolute + sim-convention relative action fields per arm side.

        Relative is ``desired - pre_step_current``: pre-step current is the EE pose
        from the most recently captured observation (the obs read this step).
        """
        cmd = action if isinstance(action, BimanualAction) else BimanualAction.from_robot_action(action)
        pre = self._obs_buf[-1] if self._obs_buf else None
        rec: dict[str, np.ndarray] = {}
        for arm in self.robot.active_arms:
            a = cmd.arm(arm)
            if a is None:
                continue
            side = _ARM_TO_SIDE[arm]
            des_pos = np.asarray(a.pose.pos, dtype=np.float64)
            des_quat = np.asarray(a.pose.quat_wxyz, dtype=np.float64)

            if pre is not None and f"{side}_ee_pos" in pre:
                cur_pos = pre[f"{side}_ee_pos"].astype(np.float64)
                cur_quat = pre[f"{side}_ee_quat"].astype(np.float64)
            else:
                cur_pos, cur_quat = des_pos, des_quat  # no obs yet -> zero relative
            err = compute_pose_error(cur_pos, cur_quat, des_pos, des_quat)

            rec[f"robot__action__poses__{side}::panda__xyz"] = des_pos.astype(np.float32)
            rec[f"robot__action__poses__{side}::panda__quat_wxyz"] = des_quat.astype(np.float32)
            rec[f"robot__action__poses__{side}::panda__xyz_relative"] = err[:3].astype(np.float32)
            rec[f"robot__action__poses__{side}::panda__axis_angle_relative"] = err[3:].astype(np.float32)
            rec[f"robot__action__grippers__{side}::panda_hand"] = np.array([a.gripper], dtype=np.float32)
        self._act_buf.append(rec)

    # ------------------------------------------------------------------
    # episode assembly + HDF5 write
    # ------------------------------------------------------------------
    def _build_episode(self, n: int, obs_buf: list, img_buf: list,
                       state_buf: list, act_buf: list) -> dict:
        """Stack the first ``n`` transitions (from the given buffers) into the
        nested demo dict. Operates on handed-off buffer snapshots so it is safe to
        run on the background writer thread while the loop fills fresh buffers."""
        def stack_dicts(dicts: list[dict]) -> dict:
            out: dict = {}
            for key in dicts[0]:
                if isinstance(dicts[0][key], dict):
                    out[key] = stack_dicts([d[key] for d in dicts])
                else:
                    out[key] = np.stack([d[key] for d in dicts], axis=0)
            return out

        obs = stack_dicts(obs_buf[:n])
        next_obs = stack_dicts(obs_buf[1:n + 1])
        action = stack_dicts(act_buf[:n])

        obs_group: dict = {"policy_dict": obs, "policy": self._concat_policy(obs)}
        if self.record_images:
            vision = self._stack_images(n, img_buf)  # {view: (n,H,W,3) uint8}
            if vision:
                # images live ONLY on obs (next_obs[t] == obs[t+1]); avoids
                # doubling the large frame storage.
                obs_group["vision_dict"] = vision

        episode: dict = {
            "action": action,
            "obs": obs_group,
            "next_obs": {"policy_dict": next_obs, "policy": self._concat_policy(next_obs)},
        }
        if self.record_states and len(state_buf) >= n + 1:
            episode["states"] = {"articulation": stack_dicts(state_buf[:n])}
            episode["next_states"] = {"articulation": stack_dicts(state_buf[1:n + 1])}
        return episode

    @staticmethod
    def _stack_images(n: int, img_buf: list) -> dict:
        """Stack the first n steps' frames per view -> {view: (n,H,W,3) uint8}.

        Only views present in ALL n steps with a consistent shape are kept (a view
        that intermittently dropped frames or changed shape is skipped, with a
        warning, rather than producing a ragged/unstackable dataset).
        """
        if n <= 0 or not img_buf:
            return {}
        views = set(img_buf[0])
        for step in img_buf[1:n]:
            views &= set(step)
        out: dict = {}
        for v in sorted(views):
            frames = [img_buf[t][v] for t in range(n)]
            shapes = {f.shape for f in frames}
            if len(shapes) != 1:
                logger.warning("Camera %s has inconsistent frame shapes %s; skipping in HDF5", v, shapes)
                continue
            out[v] = np.stack(frames, axis=0)
        dropped = (set().union(*[set(s) for s in img_buf[:n]])) - set(out)
        if dropped:
            logger.warning("Camera view(s) %s not present in all %d steps; omitted from demo", sorted(dropped), n)
        return out

    @staticmethod
    def _concat_policy(stacked_obs: dict) -> np.ndarray:
        """Flat policy vector: concat policy_dict terms in sorted-key order.

        Real-layout (not sim-bit-compatible): dims differ from sim because joints
        are 7-dof here and there is no object term. policy_dict is the source of truth.
        """
        return np.concatenate([stacked_obs[k] for k in sorted(stacked_obs)], axis=1)

    def _ensure_file(self):
        """Open/create the HDF5 (writer thread only). Does NOT touch _demo_count --
        the main thread owns that (seeded from disk in __init__, advanced at
        hand-off); the writer must not race it."""
        if self._h5 is not None:
            return
        import h5py

        if self.output_dir and not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        path = os.path.join(self.output_dir, self.filename)
        if os.path.exists(path):
            self._h5 = h5py.File(path, "a")
            logger.info("Appending to %s", path)
        else:
            self._h5 = h5py.File(path, "w")
            data = self._h5.create_group("data")
            data.attrs["total"] = 0
            data.attrs["env_args"] = json.dumps(
                {"env_name": self.env_name, "type": 2, "real": True,
                 "control_mode": getattr(self.robot.config, "control_mode", "")}
            )
            logger.info("Created %s", path)

    def _write_demo(self, episode: dict, demo_idx: int, success: bool, num_samples: int) -> None:
        """Write one demo at the reserved index (writer thread). Never overwrites:
        if demo_idx is somehow taken (e.g. a pre-existing file used those names),
        advance to the first free index."""
        self._ensure_file()
        data = self._h5["data"]
        idx = demo_idx
        while f"demo_{idx}" in data:
            idx += 1
        grp = data.create_group(f"demo_{idx}")
        grp.attrs["num_samples"] = num_samples
        grp.attrs["success"] = success

        def write(group, key, value):
            if isinstance(value, dict):
                sub = group.create_group(key)
                for k, v in value.items():
                    write(sub, k, v)
            else:
                arr = np.asarray(value)
                # Image stacks (N,H,W,3 uint8): chunk per-frame so gzip compresses
                # each frame independently (better ratio + per-frame random read).
                if arr.dtype == np.uint8 and arr.ndim == 4:
                    group.create_dataset(key, data=arr, compression="gzip",
                                         chunks=(1,) + arr.shape[1:])
                else:
                    group.create_dataset(key, data=arr, compression="gzip")

        for key, value in episode.items():
            write(grp, key, value)

        data.attrs["total"] = int(data.attrs["total"]) + num_samples
        self._h5.flush()
        logger.info("Wrote demo_%d (%d steps, success=%s)", idx, num_samples, success)
