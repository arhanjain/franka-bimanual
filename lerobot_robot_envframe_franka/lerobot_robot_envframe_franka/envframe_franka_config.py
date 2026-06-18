from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

_VALID_ARMS: tuple[str, ...] = ("l", "r")

# Base-in-env mounting transforms: real arm key -> (translation_xyz,
# quaternion_wxyz) of the arm base in the shared, world-aligned env frame.
# Quaternion is WXYZ here (IsaacLab convention); the robot converts to xyzw.
#
# The left base matches the sim scene
# (src/sim_improvement/environments/lbm/scenario_rollout_cfg.py:66-82). The real
# RIGHT arm is mounted ~180deg about env-Z from the sim value (sim yaw +47deg
# would make it asymmetric with the left's +137deg; the corrected -133deg is the
# expected mirror). Using the sim value made env +X/+Y commands drive the real
# right EE backwards and would mirror sim-trajectory replay on that arm.
# Corrected = Rz(180deg) * R_sim_right (yaw only; pitch/roll preserved).
#
# NOTE: these are hardware-tuned per arm (code-l=luigi, code-r=mario). Do NOT
# swap l/r here -- doing so rotates teleop twists and breaks the axes. The env
# observation / home symmetry questions are separate; resolve the physical arm
# assignment with scripts/misc/env_jog.py before touching these.
_DEFAULT_BASE_IN_ENV: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "l": ((-0.5937, -0.34362, -0.08484), (0.36811, 0.01027, 0.00078, 0.92973)),
    "r": ((-0.5937, 0.32962, -0.08062), (-0.39909, -0.01089, 0.01312, 0.91675)),
}


@RobotConfig.register_subclass("envframe_franka")
@dataclass
class EnvFrameFrankaConfig(RobotConfig):
    """Minimal env-frame, EE-pose-only bimanual Franka.

    Network defaults match scripts/spacemouse_teleop.sh.
    """

    l_server_ip: str = "192.168.3.11"
    l_robot_ip: str = "192.168.200.2"
    l_port: int = 18813
    r_server_ip: str = "192.168.3.10"
    r_robot_ip: str = "192.168.201.10"
    r_port: int = 18812

    active_arms: tuple[str, ...] = _VALID_ARMS

    # Per-arm base-in-env transform: arm -> (xyz, quat_wxyz). Tunable without
    # code edits if the real mounting differs from the sim scene.
    base_in_env: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(
        default_factory=lambda: {k: v for k, v in _DEFAULT_BASE_IN_ENV.items()}
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        if not self.active_arms:
            raise ValueError("active_arms must contain at least one arm: 'l' and/or 'r'.")

        invalid = [arm for arm in self.active_arms if arm not in _VALID_ARMS]
        if invalid:
            raise ValueError(f"Invalid active arm identifiers: {invalid}. Allowed: {_VALID_ARMS}.")

        self.active_arms = tuple(dict.fromkeys(self.active_arms))

        missing = [arm for arm in self.active_arms if arm not in self.base_in_env]
        if missing:
            raise ValueError(f"base_in_env missing transforms for active arms: {missing}.")
