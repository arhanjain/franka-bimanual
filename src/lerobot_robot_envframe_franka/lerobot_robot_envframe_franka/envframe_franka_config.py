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
    # "r": ((-0.5937, 0.32962, -0.08062), (0.93044, 0.01698, -0.00158, -0.36604)),
    # "l": ((-0.5937, -0.34362, -0.08484), (0.91771, 0.00781, -0.00671, 0.39712)),
    "r": ((-0.45028016754316047, 0.47616878603212737, -0.13022607619968823), (0.9277605630207296, 0.017168232629838534, -0.001211519524641021, -0.372778918009412)),
    "l": ((-0.46615435778813985, -0.4839453420803643, -0.11632933426046584), (0.9205802155596933, 0.0030479548333212396, 0.007879854069839392, 0.39046214232783283)),
}

# Default GigE camera rig, owned here so scripts don't each reinvent it (this is
# how BimanualFrankaConfig carries its `cameras` default). Same physical cameras
# as the bimanual rig: two fixed FRAMOS scene cams plus one ARV wrist cam per arm
# (added only for active_arms). IPs/serials confirmed against the bimanual stack.
#
# Capture resolution is PER CAMERA (width, height in pixels), carried in each
# entry below, matching the sim camera sizes (sim wrist = 960x600, sim scene =
# 640x480). The FRAMOS scene cams enable the color stream natively at
# (width, height) (no resize). The ARV wrist cams oversample then
# INTER_AREA-downscale to (width, height): the driver auto-picks the largest
# downscale factor (<= 8) whose capture (width*factor x height*factor) fits the
# Basler panel -- so 960x600 captures at x2 (1920x1200) and 224x224 stays x8
# (1792x1792). No fixed factor overflows the sensor anymore.
_DEFAULT_CAM_FPS = 30
# view name -> (ip, serial, width, height). Names use the canonical sim/policy
# view keys (scene_*_0, wrist_*) so the default rig's observation keys match the
# policy and the sim pkl directly -- scripts can use the default rig as-is
# instead of rebuilding cameras just to rename them.
_DEFAULT_SCENE_CAMS: dict[str, tuple[str, str, int, int]] = {
    "scene_right_0": ("192.168.0.116", "6CD146030D71", 640, 480),
    "scene_left_0": ("192.168.1.102", "6CD146030D63", 640, 480),
}
# arm -> (view name, ip, width, height)
_DEFAULT_WRIST_CAMS: dict[str, tuple[str, str, int, int]] = {
    # "l": ("wrist_left_minus", "192.168.1.138", 960, 600),
    # "l": ("wrist_left_plus", "192.168.1.139", 960, 600),
    "l": ("wrist_left_plus", "192.168.1.139", 960, 600),
    "r": ("wrist_right_minus", "192.168.0.142", 960, 600),
    # "r+": ("wrist_right_plus", "192.168.0.143", 960, 600),
}


def _default_cameras(active_arms: tuple[str, ...]) -> dict:
    """Build the default {view: CameraConfig} map: both scene cams + the wrist
    cam for each active arm, each at its own (sim-matching) capture resolution.
    Camera-config imports are lazy/guarded so pose-only sessions
    (enable_cameras=False) never pull Aravis/librealsense."""
    from lerobot_camera_arv import ArvCameraConfig
    from lerobot_camera_framos import FramosCameraConfig

    cams: dict = {}
    for view, (ip, sn, w, h) in _DEFAULT_SCENE_CAMS.items():
        cams[view] = FramosCameraConfig(
            name=f"framos_{sn}", ip=ip, serial_number=sn,
            fps=_DEFAULT_CAM_FPS, width=w, height=h,
        )
    for arm in active_arms:
        view, ip, w, h = _DEFAULT_WRIST_CAMS[arm]
        cams[view] = ArvCameraConfig(
            name=f"arv_{ip}", ip=ip,
            fps=_DEFAULT_CAM_FPS, width=w, height=h,
        )
    return cams


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

    # Schunk WSG gripper IPs (GCL over TCP, same driver as BimanualFranka).
    # l = mario gripper, r = luigi gripper (README hardware map). Only used when
    # enable_grippers is True; pose-only sessions leave grippers untouched.
    enable_grippers: bool = False
    l_gripper_ip: str = "192.168.2.21"
    r_gripper_ip: str = "192.168.2.20"

    active_arms: tuple[str, ...] = _VALID_ARMS

    # Per-arm base-in-env transform: arm -> (xyz, quat_wxyz). Tunable without
    # code edits if the real mounting differs from the sim scene.
    base_in_env: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(
        default_factory=lambda: {k: v for k, v in _DEFAULT_BASE_IN_ENV.items()}
    )

    # Cameras, keyed by observation/view name (same Arv/Framos configs as
    # BimanualFrankaConfig). When enable_cameras is True and `cameras` is left
    # empty, __post_init__ populates the default rig (both scene cams + one wrist
    # per active arm) via _default_cameras. Pass an explicit dict to override the
    # mapping (e.g. policy-view names), or set enable_cameras=False for pose-only
    # sessions (home/circle/replay) that need no vision.
    enable_cameras: bool = True
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
    cart_gain: float = 6.0
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

        # Populate the default rig when cameras are enabled and none were given
        # explicitly; an explicit dict (e.g. policy-view names) is left untouched.
        if self.enable_cameras and not self.cameras:
            self.cameras = _default_cameras(self.active_arms)
        elif not self.enable_cameras:
            self.cameras = {}

        camera_names = [str(getattr(c, "name", "")) for c in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")
