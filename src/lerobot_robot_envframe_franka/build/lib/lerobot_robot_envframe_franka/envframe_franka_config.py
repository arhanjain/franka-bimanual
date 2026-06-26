from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

_VALID_ARMS: tuple[str, ...] = ("l", "r")
_VALID_CONTROL_MODES: tuple[str, ...] = ("twist", "joint_ik")

# Sim-matching joint controller gains for joint_ik mode. The sim's
# IMPLICIT_PANDA actuators use a uniform 400 N*m/rad joint-impedance stiffness;
# matching it on the FR3's internal joint controller is the whole point of the
# joint_ik path (DLS-IK target -> stiff joint PD, just like sim). Tune from here.
_SIM_JOINT_STIFFNESS: tuple[float, ...] = (400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0)
# (velocity, accel, jerk) scale for the joint-position stream's Ruckig planner.
# Conservative for bring-up; the 100 Hz q_des stream is already smooth, so this
# mostly bounds the transient when the held target jumps at the 10 Hz action rate.
_JOINT_IK_RELATIVE_DYNAMICS: tuple[float, ...] = (0.4, 0.25, 0.15)

# Base-in-env mounting transforms: real arm key -> (translation_xyz,
# quaternion_wxyz) of the arm base in the shared, world-aligned env frame.
# Quaternion is WXYZ here (IsaacLab convention); the robot converts to xyzw.
#
# Identity (positions = scenario_rollout_cfg.py:66-82; IP defaults below agree):
#   code l = sim left_panda  = mario NUC (192.168.3.10) = env -y (-0.34362)
#   code r = sim right_panda = luigi NUC (192.168.3.11) = env +y (+0.32962)
# Env axes: +x toward workspace, +y, +z up.
#
# POSITIONS are the verbatim sim placements. ORIENTATIONS are the sim
# quaternions RE-AIMED -90deg (90 CW viewed from above) about env +z, so each
# base +x faces env +x (the workspace) instead of the sim scene's +y-ish
# mounting. Deliberate divergence from the raw sim -- to re-sync, re-take the
# sim quats and re-apply Rz(-90deg) about env z.
# NOTE: re-aim not yet re-confirmed on hardware via scripts/misc/env_jog.py.

# wxyz (sim orientation re-aimed -90deg about env z; see above)
_DEFAULT_BASE_IN_ENV: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "r": ((-0.5937, 0.32962, -0.08062), (0.93044, 0.01698, -0.00158, -0.36604)),
    "l": ((-0.5937, -0.34362, -0.08484), (0.91771, 0.00781, -0.00671, 0.39712)),
}


@RobotConfig.register_subclass("envframe_franka")
@dataclass
class EnvFrameFrankaConfig(RobotConfig):
    """Minimal env-frame, EE-pose-only bimanual Franka.

    Network defaults match scripts/spacemouse_teleop.sh.
    """

    # Arm naming follows SIM convention (left/right as seen facing the robots):
    #   l = LEFT = mario NUC (192.168.3.10, robot 192.168.201.10, port 18812)
    #   r = RIGHT = luigi NUC (192.168.3.11, robot 192.168.200.2, port 18813)
    # This makes code l/r match sim left/right and the base_in_env below correct.
    l_server_ip: str = "192.168.3.10"
    l_robot_ip: str = "192.168.201.10"
    l_port: int = 18812
    r_server_ip: str = "192.168.3.11"
    r_robot_ip: str = "192.168.200.2"
    r_port: int = 18813

    active_arms: tuple[str, ...] = _VALID_ARMS

    # Per-arm base-in-env transform: arm -> (xyz, quat_wxyz). Tunable without
    # code edits if the real mounting differs from the sim scene.
    base_in_env: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(
        default_factory=lambda: {k: v for k, v in _DEFAULT_BASE_IN_ENV.items()}
    )

    # Optional cameras (empty by default -- pose-only teleop/replay needs none).
    # Keyed by observation name; same Arv/Framos configs as BimanualFrankaConfig.
    # A vision-policy inference client populates this (see scripts factory).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Actuation path for the (identical) absolute-EE-pose action interface:
    #   "joint_ik" (DEFAULT) -- workstation runs the sim DLS-IK resolved-rate loop
    #                 at ik_hz and streams joint velocities tracked by the FR3
    #                 joint controller. Matches the sim "DLS-IK -> joint" mapping
    #                 so sim/real can be compared/tuned (gains: cart_gain,
    #                 joint_stiffness). This is the primary path.
    #   "twist"    -- Cartesian-velocity PD tracking via franky (original path).
    control_mode: str = "joint_ik"
    # joint_ik knobs (ignored in twist mode).
    ik_hz: float = 100.0                 # inner DLS-IK loop rate (sim policy=10 Hz; we oversample)
    dls_lambda: float = 0.01             # DLS damping; sim default
    # Cartesian resolved-rate gain (1/s): commanded joint velocity = DLS(J, dx) *
    # cart_gain. This is the loop gain -- it sets how fast the EE error is driven
    # to zero (v_ee ~= cart_gain * pos_err), decoupled from ik_hz. Too high =
    # overshoot + springy/noisy motion (driving the whole error in one tick is
    # gain ~= ik_hz = 100, which oscillates); the repo's prototype_circle used
    # 4.0. Start low and raise until tracking is crisp without ringing.
    cart_gain: float = 4.0
    max_joint_vel: float = 1.0           # rad/s; per-joint commanded-velocity clamp (singularity/safety guard)
    joint_stiffness: tuple[float, ...] = _SIM_JOINT_STIFFNESS
    joint_ik_relative_dynamics: tuple[float, ...] = _JOINT_IK_RELATIVE_DYNAMICS

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        if self.control_mode not in _VALID_CONTROL_MODES:
            raise ValueError(
                f"Invalid control_mode {self.control_mode!r}. Allowed: {_VALID_CONTROL_MODES}."
            )

        if not self.active_arms:
            raise ValueError("active_arms must contain at least one arm: 'l' and/or 'r'.")

        invalid = [arm for arm in self.active_arms if arm not in _VALID_ARMS]
        if invalid:
            raise ValueError(f"Invalid active arm identifiers: {invalid}. Allowed: {_VALID_ARMS}.")

        self.active_arms = tuple(dict.fromkeys(self.active_arms))

        missing = [arm for arm in self.active_arms if arm not in self.base_in_env]
        if missing:
            raise ValueError(f"base_in_env missing transforms for active arms: {missing}.")

        camera_names = [str(getattr(c, "name", "")) for c in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")
