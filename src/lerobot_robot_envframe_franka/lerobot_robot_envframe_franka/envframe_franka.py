"""Minimal env-frame, EE-pose-only bimanual Franka robot for LeRobot.

Actions and observations are absolute EE poses expressed in the sim **env frame**
(world-aligned axes, shared origin), keyed ``{arm}_{x,y,z,qx,qy,qz,qw}`` per
active arm (quaternion xyzw). The action interface is identical across two
actuation modes selected by ``config.control_mode``:

- ``"twist"`` (default): ``send_action`` converts the env-frame target into each
  arm's base frame and tracks it with a base-frame Cartesian-velocity PD twist
  streamed to franky (``CartesianVelocityMotion``).
- ``"joint_ik"``: ``send_action`` only stores the per-arm base-frame target pose
  (held between calls); a background thread runs the sim's DLS-IK law at
  ``config.ik_hz`` against the live measured state and streams the resulting
  joint step as a joint *velocity* (``JointVelocityMotion``) tracked by the FR3
  joint controller. The redundancy resolution is sim's exact
  ``Jᵀ(JJᵀ+λ²I)⁻¹``, so the joint-space trajectory matches sim; the action rate
  (e.g. 10 Hz) is decoupled from the inner control rate, and ``joint_stiffness``
  tunes how the FR3 tracks it. (A joint-velocity stream is used, not a streamed
  joint-position target: franky runs each move through its controller and bare
  ``JointMotion`` has no validity window, so streaming it would leave the arm
  unheld between moves and it would sag. See franka_link.py.)

``get_observation`` transforms the measured base-frame ``O_T_EE`` back into the
env frame in both modes.
"""

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from .actions import ArmAction, BimanualAction
from .diffik import compute_pose_error, dls_delta_q
from .envframe_franka_config import EnvFrameFrankaConfig
from .franka_jacobian import zero_jacobian
from .franka_link import MultiRobotWrapper

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_S = 10.0
_IK_THREAD_JOIN_TIMEOUT_S = 2.0
_CAMERA_READ_TIMEOUT_MS = 5.0
IMAGE_CHANNELS = 3


def _cam_status(view: str, cam, ok: bool) -> None:
    """One-line per-camera connect status on stderr, matching the arm's ✓/✗ line.
    The full failure reason stays in the accompanying logger.warning."""
    ip = getattr(getattr(cam, "_config", None), "ip", "") or getattr(cam, "_ip", "")
    where = f" @ {ip}" if ip else ""
    mark = "✓ connected camera" if ok else "✗ failed to connect camera"
    sys.stderr.write(f"{mark} {view}{where}\n")
    sys.stderr.flush()


def _make_camera(cfg):
    """Build an Arv/Framos camera from its config. Imported lazily so pose-only
    sessions (and machines without Aravis/librealsense) never pull these deps."""
    from lerobot_camera_arv import ArvCamera, ArvCameraConfig
    from lerobot_camera_framos import FramosCamera, FramosCameraConfig

    ctors = {ArvCameraConfig: ArvCamera, FramosCameraConfig: FramosCamera}
    cls = ctors.get(type(cfg))
    if cls is None:
        raise TypeError(f"Unsupported camera config: {type(cfg).__name__}")
    return cls(cfg)

EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw")
# Action keys add gripper per arm so the action space matches the sim
# LBM-Scenario-ImplicitIK space (per-arm 7-pose + 1 gripper). The gripper is
# ACCEPTED but not actuated yet (no gripper hardware in the env-frame stack).
ACTION_AXIS_KEYS: tuple[str, ...] = (*EE_AXIS_KEYS, "gripper")

# Cartesian PD tracking gains + velocity clamps (mirror bimanual_franka EE mode +
# safety.py limits). The pose target is tracked by a base-frame twist command.
EE_PD_KP, EE_PD_KD = 2.0, 0.1
EE_LINEAR_VELOCITY_MAX = 0.30   # m/s
EE_ANGULAR_VELOCITY_MAX = 1.20  # rad/s

# Joint-space PD gains + velocity clamp for home() (mirror bimanual_franka joint
# mode + safety.py limit). Joint control is used ONLY by home(); the action
# interface stays EE-pose. Homing in joint space pins all 7 DOF, so a symmetric
# joint target yields physically symmetric arms (EE-pose homing leaves the
# redundant elbow/wrist free and cannot guarantee that).
JOINT_PD_KP, JOINT_PD_KD = 2.0, 0.1
JOINT_VELOCITY_MAX = 2.0  # rad/s


def _clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * (max_norm / n) if n > max_norm else v


def _quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> WXYZ quaternion (sim/diffik convention)."""
    x, y, z, w = Rotation.from_matrix(R).as_quat()  # scipy returns xyzw
    return np.array([w, x, y, z], dtype=np.float64)


def _make_gripper(name: str, ip: str):
    """Schunk WSG gripper (GCL over TCP). Imported lazily/from the bimanual
    package so pose-only sessions never open a gripper socket. Same driver +
    [0,1]-vs-GRIPPER_TRUE_MAX_MM normalization as BimanualFranka."""
    from lerobot_robot_bimanual_franka.wsg import WSG

    return WSG(name=name, TCP_IP=ip, do_print=False)


class EnvFrameFranka(Robot):
    config_class = EnvFrameFrankaConfig
    name = "envframe_franka"

    def __init__(self, config: EnvFrameFrankaConfig):
        super().__init__(config)
        self.config = config
        self.active_arms = config.active_arms
        self.robot_manager = MultiRobotWrapper()

        # Optional cameras (empty unless a vision client configured them). Built
        # lazily so a pose-only session never imports Aravis/librealsense.
        self.cameras = {n: _make_camera(c) for n, c in config.cameras.items()}
        self._camera_pool = (
            ThreadPoolExecutor(max_workers=len(self.cameras)) if self.cameras else None
        )

        # Optional WSG grippers (one per active arm) when enabled; the socket is
        # opened here (WSG.__init__ connects), homed in connect(), closed in
        # disconnect(). Empty for pose-only sessions.
        self.grippers: dict = {}
        if config.enable_grippers:
            for arm in self.active_arms:
                self.grippers[arm] = _make_gripper(arm, getattr(config, f"{arm}_gripper_ip"))

        # Precompute per-arm base-in-env transform (translation + scipy rotation).
        # Config stores quaternion as wxyz; scipy wants xyzw.
        self._base: dict[str, tuple[Rotation, np.ndarray]] = {}
        for arm in self.active_arms:
            (px, py, pz), (qw, qx, qy, qz) = config.base_in_env[arm]
            self._base[arm] = (
                Rotation.from_quat([qx, qy, qz, qw]),
                np.array([px, py, pz], dtype=np.float64),
            )

        # joint_ik mode: the inner resolved-rate loop tracks a held base-frame
        # target pose per arm. send_action only updates _targets (under
        # _target_lock); _ik_thread runs the IK at config.ik_hz.
        self._joint_ik = config.control_mode == "joint_ik"
        self._targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # arm -> (pos_base, quat_base_wxyz)
        self._target_lock = threading.Lock()
        self._ik_thread: threading.Thread | None = None
        self._ik_stop = threading.Event()

    # ------------------------------------------------------------------
    # LeRobot Robot contract
    # ------------------------------------------------------------------
    @property
    def _arm_features(self) -> dict[str, type]:
        return {f"{arm}_{key}": float for arm in self.active_arms for key in EE_AXIS_KEYS}

    @property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        out: dict[str, tuple[int, int, int]] = {}
        for n, cam in self.cameras.items():
            if cam.height is None or cam.width is None:
                raise RuntimeError(f"Camera '{n}' does not report height/width")
            out[n] = (int(cam.height), int(cam.width), IMAGE_CHANNELS)
        return out

    @property
    def _gripper_features(self) -> dict[str, type]:
        return {f"{arm}_gripper": float for arm in self.grippers}

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        # EE pose per arm + gripper width (when enabled) + any configured cameras.
        return {**self._arm_features, **self._gripper_features, **self._camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        # Action carries gripper per arm to match the sim LBM-ImplicitIK space.
        return {f"{arm}_{key}": float for arm in self.active_arms for key in ACTION_AXIS_KEYS}

    @property
    def is_connected(self) -> bool:
        return self.robot_manager.num_alive == len(self.active_arms)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        # joint_ik passes the sim-matching joint impedance + JP dynamics so the
        # FR3's internal joint controller behaves like sim's implicit PD; twist
        # mode leaves the driver on its conservative defaults.
        extra = {}
        if self._joint_ik:
            extra = dict(
                stiffness=tuple(self.config.joint_stiffness),
                dynamics=tuple(self.config.joint_ik_relative_dynamics),
            )
        try:
            for n, cam in self.cameras.items():
                try:
                    cam.connect()
                    _cam_status(n, cam, ok=True)
                except Exception as e:
                    _cam_status(n, cam, ok=False)
                    logger.warning("Camera %s failed to connect: %s", n, e)
            for arm in self.active_arms:
                self.robot_manager.add_robot(
                    arm,
                    getattr(self.config, f"{arm}_server_ip"),
                    getattr(self.config, f"{arm}_robot_ip"),
                    getattr(self.config, f"{arm}_port"),
                    **extra,
                )
                self.robot_manager.current_kinematic_state(arm, timeout_s=_CONNECT_TIMEOUT_S)
            # Reference the grippers (blocking HOME) so width commands are valid.
            for arm, g in self.grippers.items():
                try:
                    g.home()
                except Exception as e:
                    logger.warning("Gripper %s failed to home: %s", arm, e)
        except Exception:
            self._close_grippers()
            self.robot_manager.shutdown()
            raise

    def _close_grippers(self) -> None:
        for arm, g in self.grippers.items():
            try:
                g.close()
            except Exception as e:
                logger.debug("Gripper %s close error: %s", arm, e)

    def disconnect(self) -> None:
        self._stop_ik_thread()
        if self._camera_pool is not None:
            self._camera_pool.shutdown(wait=False)
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.debug("Camera disconnect error: %s", e)
        self._close_grippers()
        self.robot_manager.shutdown()

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Kick off camera reads in parallel with the (blocking) kinematic query.
        cam_futs = {}
        if self._camera_pool is not None:
            cam_futs = {
                n: self._camera_pool.submit(cam.async_read, _CAMERA_READ_TIMEOUT_MS)
                for n, cam in self.cameras.items()
            }

        kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        obs: RobotObservation = {}
        for arm in self.active_arms:
            _, _, T_eb, _ = kin[arm]
            p_ee, q_ee = self._base_to_env(arm, T_eb)
            for key, val in zip(EE_AXIS_KEYS, (*p_ee, *q_ee)):
                obs[f"{arm}_{key}"] = float(val)

        # Gripper finger gap in METERS (last cached POS? mm /1000; never blocks),
        # matching the action units and sim's gripper_pos convention.
        for arm, g in self.grippers.items():
            pos = g.position
            obs[f"{arm}_gripper"] = (0.0 if pos is None else pos) / 1000.0

        for n, fut in cam_futs.items():
            try:
                obs[n] = fut.result()
            except Exception as e:
                logger.warning("Camera %s read failed: %s", n, e)
                blank = getattr(self.cameras[n], "blank_frame", None)
                obs[n] = blank() if callable(blank) else np.zeros(self._camera_features[n], dtype=np.uint8)
        return obs

    def send_action(self, action: RobotAction | BimanualAction) -> RobotAction:
        """Command an absolute env-frame pose (+ gripper) per arm.

        Accepts a typed ``BimanualAction`` (preferred) or a LeRobot float dict
        (``{arm}_{x,y,z,qx,qy,qz,qw[,gripper]}``, quats XYZW). Returns the LeRobot
        dict actually dispatched.

        Grippers (when ``enable_grippers``): the per-arm gripper field is the
        commanded finger gap **in meters** (sim units), passed to the WSG as mm
        (x1000). The WSG driver clamps to its usable 10..100 mm stroke, so 0.0 m ->
        closed (~10 mm) and 0.1 m -> open (100 mm). Measured hardware max is
        ~0.109 m; the driver caps the open end at 0.1 m. Non-blocking; the WSG
        sender coalesces repeats. If grippers are disabled the field is ignored.
        """
        cmd = action if isinstance(action, BimanualAction) else BimanualAction.from_robot_action(action)
        self._actuate_grippers(cmd)
        if self._joint_ik:
            self._send_action_joint_ik(cmd)
        else:
            self._send_action_twist(cmd)
        return cmd.to_robot_action()

    def _actuate_grippers(self, cmd: BimanualAction) -> None:
        """Drive each active WSG gripper from the action's gripper field (meters)."""
        for arm, g in self.grippers.items():
            a = cmd.arm(arm)
            if a is None:
                continue
            width_mm = float(a.gripper) * 1000.0  # meters -> mm; WSG.move clamps to 10..100 mm
            try:
                g.move(width_mm, blocking=False)
            except Exception as e:
                logger.warning("Gripper %s move failed: %s", arm, e)

    def send_bimanual_action(self, action: BimanualAction) -> BimanualAction:
        """Typed entry point; identical effect to ``send_action`` of the struct."""
        self.send_action(action)
        return action

    def _arm_targets_base(self, cmd: BimanualAction):
        """Per active+present arm: env-frame Pose -> base-frame 4x4 target."""
        out: dict[str, np.ndarray] = {}
        for arm in self.active_arms:
            a: ArmAction | None = cmd.arm(arm)
            if a is None:
                continue
            out[arm] = self._env_to_base(arm, a.pose.pos, a.pose.quat_xyzw)
        return out

    def _send_action_twist(self, cmd: BimanualAction) -> None:
        # Track the absolute env-frame pose target with a base-frame PD twist.
        targets = self._arm_targets_base(cmd)
        kin = self.robot_manager.current_kinematic_state_batch(list(targets))
        twists: dict[str, list] = {}
        for arm, T_tb in targets.items():
            _, _, T_eb, twist = kin[arm]                     # current pose + twist (base)
            pos_err = T_tb[:3, 3] - T_eb[:3, 3]
            rot_err = Rotation.from_matrix(T_tb[:3, :3] @ T_eb[:3, :3].T).as_rotvec()
            v = _clamp_norm(EE_PD_KP * pos_err - EE_PD_KD * twist[:3], EE_LINEAR_VELOCITY_MAX)
            w = _clamp_norm(EE_PD_KP * rot_err - EE_PD_KD * twist[3:], EE_ANGULAR_VELOCITY_MAX)
            twists[arm] = [*v.tolist(), *w.tolist()]
        self.robot_manager.move_twist_batch(twists)

    def _send_action_joint_ik(self, cmd: BimanualAction) -> None:
        # Only update the held base-frame target pose per arm; the inner IK loop
        # (started lazily on the first action) tracks it at config.ik_hz. The
        # base-frame target pose is what DLS-IK consumes -- the Jacobian and the
        # measured O_T_EE the loop reads are both base-frame, so all the env->base
        # framing happens once here, not every inner tick.
        targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm, T_tb in self._arm_targets_base(cmd).items():
            targets[arm] = (T_tb[:3, 3].copy(), _quat_wxyz_from_matrix(T_tb[:3, :3]))
        with self._target_lock:
            self._targets = targets
        self._ensure_ik_thread()

    # ------------------------------------------------------------------
    # joint_ik inner control loop
    # ------------------------------------------------------------------
    def _ensure_ik_thread(self) -> None:
        if self._ik_thread is not None and self._ik_thread.is_alive():
            return
        self._ik_stop.clear()
        self._ik_thread = threading.Thread(target=self._ik_loop, name="envframe-ik", daemon=True)
        self._ik_thread.start()

    def _stop_ik_thread(self) -> None:
        if self._ik_thread is None:
            return
        self._ik_stop.set()
        self._ik_thread.join(timeout=_IK_THREAD_JOIN_TIMEOUT_S)
        self._ik_thread = None
        # Bring the joint-velocity stream to rest (zero-velocity command holds).
        try:
            self.robot_manager.stop_all_joint_motion()
        except Exception:
            logger.exception("error stopping joint motion")

    def _ik_loop(self) -> None:
        """Resolved-rate Cartesian tracking at config.ik_hz via the sim DLS map.

        Per tick, per arm: read (q, O_T_EE), build the base-frame geometric
        Jacobian from q (franky's model.zero_jacobian is broken here), form the
        base-frame pose error ``dx`` against the HELD target, and command a joint
        velocity ``dq = DLS(J, cart_gain * dx)`` clamped to ``max_joint_vel``.
        ``cart_gain`` is the loop gain (1/s) setting how fast the EE error is
        driven to zero (``v_ee ~= cart_gain * dx``); the DLS pseudo-inverse is
        sim's exact ``Jᵀ(JJᵀ+λ²I)⁻¹`` so the redundancy resolution matches sim.

        Commanded via the JointVelocityMotion stream home() uses, NOT a streamed
        JointMotion: franky runs each move through its controller, and bare
        JointMotion is a point-to-point Ruckig move with no validity window --
        streaming it at 100 Hz trips "multiple motions" every tick, nothing holds
        the arm, and it sags. A velocity motion's 100 ms window holds at
        convergence (zero-vel), so the arm is always under active control.
        """
        period = 1.0 / self.config.ik_hz
        names = list(self.active_arms)
        v_max = self.config.max_joint_vel
        ticks = 0
        sends = 0
        hb_t = time.perf_counter()
        dbg = {arm: (0.0, 0.0, 0.0, 0.0) for arm in names}  # arm -> (pos_err_m, dq, |v|max, |J|)
        while not self._ik_stop.is_set():
            t0 = time.perf_counter()
            with self._target_lock:
                targets = self._targets
            if targets:
                try:
                    ik = self.robot_manager.current_ik_state_batch(names)
                    cmds: dict[str, list] = {}
                    for arm in names:
                        if arm not in targets:
                            continue
                        q, T_eb = ik[arm]
                        pos_des, quat_des = targets[arm]
                        pos_cur = T_eb[:3, 3]
                        quat_cur = _quat_wxyz_from_matrix(T_eb[:3, :3])
                        # franky's model.zero_jacobian is broken here (returns 0),
                        # so compute the base-frame geometric Jacobian from q,
                        # anchored on the measured EE point for consistency.
                        J = zero_jacobian(q, pos_cur)
                        # Resolved-rate: command a Cartesian velocity proportional
                        # to the pose error (v_ee = cart_gain * dx), then map it to
                        # joint velocity through the sim DLS pseudo-inverse. cart_gain
                        # is the loop gain (1/s) -- NOT ik_hz, which would try to null
                        # the whole error every tick (gain ~100) and ring (springy).
                        dx = compute_pose_error(pos_cur, quat_cur, pos_des, quat_des)
                        v_ee = self.config.cart_gain * dx
                        dq = dls_delta_q(J, v_ee, lam=self.config.dls_lambda)
                        dq_cmd = np.clip(dq, -v_max, v_max)
                        cmds[arm] = dq_cmd.tolist()
                        dbg[arm] = (
                            float(np.linalg.norm(pos_des - pos_cur)),
                            float(np.linalg.norm(dq)),           # |dq| (rad/s) before clamp
                            float(np.max(np.abs(dq_cmd))),
                            float(np.linalg.norm(J)),            # |J|: 0 => zero Jacobian (bad read)
                        )
                    if cmds:
                        self.robot_manager.move_joint_velocity_batch(cmds)
                        sends += 1
                except Exception:
                    # Never let a transient RPyC/read error kill the loop; warn
                    # and retry next tick (the driver also self-recovers known
                    # recoverable franky faults).
                    logger.exception("IK loop tick failed; retrying next tick")
            ticks += 1
            # Heartbeat ~1 Hz: achieved tick rate + per-arm pose error and the
            # peak commanded joint velocity. Diagnoses "not moving": err~0 means
            # the target equals the current pose (nothing commanded); err large
            # with |v|~0 means the command is being dropped/clamped.
            now = time.perf_counter()
            if now - hb_t >= 1.0:
                # rate = ticks / (now - hb_t)
                # summary = "  ".join(
                #     f"{a}: perr={dbg[a][0]*1000:.1f}mm dq={dbg[a][1]:.4f} "
                #     f"|v|max={dbg[a][2]:.3f} |J|={dbg[a][3]:.2f}"
                #     for a in names
                # )
                # logger.info("ik_loop %.0f Hz (sends=%d/%d)  %s", rate, sends, ticks, summary)
                ticks = 0
                sends = 0
                hb_t = now
            dt = time.perf_counter() - t0
            if dt < period:
                self._ik_stop.wait(period - dt)

    def home(
        self,
        targets_q: dict[str, np.ndarray],
        max_time_s: float = 8.0,
        tol_rad: float = 0.05,
        fps: float = 30.0,
    ) -> bool:
        """Drive the active arms to per-arm joint targets via joint-velocity PD.

        Joint space (not EE) so every DOF is pinned: a mirror-symmetric joint
        target produces physically symmetric arms. Returns True if every arm
        reached ``tol_rad`` (L-inf, per joint) before ``max_time_s``.
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        targets = {a: np.asarray(q, dtype=np.float64) for a, q in targets_q.items() if a in self.active_arms}
        if not targets:
            return True
        names = list(targets)

        period = 1.0 / fps
        deadline = time.perf_counter() + max_time_s
        converged = False
        try:
            while time.perf_counter() < deadline:
                t0 = time.perf_counter()
                kin = self.robot_manager.current_kinematic_state_batch(names)
                cmds: dict[str, list] = {}
                for arm in names:
                    q, dq = kin[arm][0], kin[arm][1]
                    v = JOINT_PD_KP * (targets[arm] - q) - JOINT_PD_KD * dq
                    cmds[arm] = _clamp_norm(v, JOINT_VELOCITY_MAX).tolist()
                self.robot_manager.move_joint_velocity_batch(cmds)

                if max(float(np.max(np.abs(targets[arm] - kin[arm][0]))) for arm in names) < tol_rad:
                    converged = True
                    break
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            self.robot_manager.stop_all_joint_motion()
        return converged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def current_ee_pose_env(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-arm current EE pose in the env frame: {arm: (xyz, quat_xyzw)}.

        Intended for seeding a teleoperator's integrated pose before the loop.
        """
        kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in self.active_arms:
            _, _, T_eb, _ = kin[arm]
            out[arm] = self._base_to_env(arm, T_eb)
        return out

    def _env_to_base(self, arm: str, p_te: np.ndarray, q_te_xyzw: np.ndarray) -> np.ndarray:
        """env-frame target (pos, quat xyzw) -> base-frame 4x4 homogeneous matrix."""
        R_be, p_be = self._base[arm]
        R_be_inv = R_be.inv()
        p_tb = R_be_inv.apply(p_te - p_be)
        R_tb = R_be_inv * Rotation.from_quat(q_te_xyzw)
        T = np.eye(4)
        T[:3, :3] = R_tb.as_matrix()
        T[:3, 3] = p_tb
        return T

    def _base_to_env(self, arm: str, T_eb: np.ndarray):
        """base-frame measured 4x4 pose -> env-frame (pos, quat xyzw)."""
        R_be, p_be = self._base[arm]
        p_eb = T_eb[:3, 3]
        R_eb = Rotation.from_matrix(T_eb[:3, :3])
        p_ee = p_be + R_be.apply(p_eb)
        R_ee = R_be * R_eb
        return p_ee, R_ee.as_quat()
