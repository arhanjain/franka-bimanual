"""Minimal env-frame, EE-pose-only bimanual Franka robot for LeRobot.

Actions and observations are absolute EE poses expressed in the sim **env frame**
(world-aligned axes, shared origin), keyed ``{arm}_{x,y,z,qx,qy,qz,qw}`` per
active arm (quaternion xyzw). The robot is stateless: ``send_action`` transforms
the env-frame target into each arm's base frame and streams a franky absolute
``CartesianMotion``; ``get_observation`` transforms the measured base-frame
``O_T_EE`` back into the env frame.
"""

import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from .envframe_franka_config import EnvFrameFrankaConfig
from .franka_link import MultiRobotWrapper

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_S = 10.0

EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw")

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


class EnvFrameFranka(Robot):
    config_class = EnvFrameFrankaConfig
    name = "envframe_franka"

    def __init__(self, config: EnvFrameFrankaConfig):
        super().__init__(config)
        self.config = config
        self.active_arms = config.active_arms
        self.robot_manager = MultiRobotWrapper()

        # Precompute per-arm base-in-env transform (translation + scipy rotation).
        # Config stores quaternion as wxyz; scipy wants xyzw.
        self._base: dict[str, tuple[Rotation, np.ndarray]] = {}
        for arm in self.active_arms:
            (px, py, pz), (qw, qx, qy, qz) = config.base_in_env[arm]
            self._base[arm] = (
                Rotation.from_quat([qx, qy, qz, qw]),
                np.array([px, py, pz], dtype=np.float64),
            )

    # ------------------------------------------------------------------
    # LeRobot Robot contract
    # ------------------------------------------------------------------
    @property
    def _arm_features(self) -> dict[str, type]:
        return {f"{arm}_{key}": float for arm in self.active_arms for key in EE_AXIS_KEYS}

    @property
    def observation_features(self) -> dict[str, type]:
        return dict(self._arm_features)

    @property
    def action_features(self) -> dict[str, type]:
        return dict(self._arm_features)

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
        try:
            for arm in self.active_arms:
                self.robot_manager.add_robot(
                    arm,
                    getattr(self.config, f"{arm}_server_ip"),
                    getattr(self.config, f"{arm}_robot_ip"),
                    getattr(self.config, f"{arm}_port"),
                )
                self.robot_manager.current_kinematic_state(arm, timeout_s=_CONNECT_TIMEOUT_S)
        except Exception:
            self.robot_manager.shutdown()
            raise

    def disconnect(self) -> None:
        self.robot_manager.shutdown()

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        obs: RobotObservation = {}
        for arm in self.active_arms:
            _, _, T_eb, _ = kin[arm]
            p_ee, q_ee = self._base_to_env(arm, T_eb)
            for key, val in zip(EE_AXIS_KEYS, (*p_ee, *q_ee)):
                obs[f"{arm}_{key}"] = float(val)
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        # Track the absolute env-frame pose target with a base-frame PD twist.
        kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        twists: dict[str, list] = {}
        for arm in self.active_arms:
            p_te = np.array([action[f"{arm}_{k}"] for k in ("x", "y", "z")], dtype=np.float64)
            q_te = np.array([action[f"{arm}_{k}"] for k in ("qx", "qy", "qz", "qw")], dtype=np.float64)
            T_tb = self._env_to_base(arm, p_te, q_te)        # target in base frame
            _, _, T_eb, twist = kin[arm]                     # current pose + twist (base)

            pos_err = T_tb[:3, 3] - T_eb[:3, 3]
            rot_err = Rotation.from_matrix(T_tb[:3, :3] @ T_eb[:3, :3].T).as_rotvec()
            v = _clamp_norm(EE_PD_KP * pos_err - EE_PD_KD * twist[:3], EE_LINEAR_VELOCITY_MAX)
            w = _clamp_norm(EE_PD_KP * rot_err - EE_PD_KD * twist[3:], EE_ANGULAR_VELOCITY_MAX)
            twists[arm] = [*v.tolist(), *w.tolist()]
        self.robot_manager.move_twist_batch(twists)
        return action

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
