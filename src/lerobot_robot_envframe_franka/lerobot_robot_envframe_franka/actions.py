"""Typed action structs mirroring the sim ``LBM-Scenario-ImplicitIK`` action space.

A trajectory replay, a teleop leader, or a policy all build the SAME explicit
type instead of an ad-hoc ``{arm}_{axis}`` float dict, so the type makes clear
WHICH action is being commanded (absolute env-frame pose + gripper, per arm).

Sim action layout (verified against the live ActionManager:
``dual_arm_ik_action`` = 14, then ``left_panda_gripper`` = 1, ``right_panda_gripper`` = 1):

    [ 0: 3]  left  xyz             absolute, env frame, meters
    [ 3: 7]  left  quat  (wxyz)
    [ 7:10]  right xyz
    [10:14]  right quat  (wxyz)
    [14]     left  gripper
    [15]     right gripper

The per-arm 7-vector is exactly IsaacLab's ``DifferentialIKController`` absolute
pose command (``command_type="pose"``, ``use_relative_mode=False``): ``(x, y, z,
qw, qx, qy, qz)``.

Conventions held by these structs:
- Quaternions are WXYZ (IsaacLab / sim / HDF5 convention). ``EnvFrameFranka``
  converts to XYZW at its franky boundary; use ``Pose.from_quat_xyzw`` /
  ``Pose.quat_xyzw`` to cross that boundary explicitly.
- Positions are in the sim env frame (world-aligned axes, shared origin).
- ``gripper`` is the raw sim command scalar. EnvFrameFranka currently ACCEPTS it
  but does not actuate it (no gripper hardware in the env-frame stack yet).
- ``left`` -> robot arm ``l`` (mario / env -y / sim ``left_panda``);
  ``right`` -> robot arm ``r`` (luigi / env +y / sim ``right_panda``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Flat sim-vector layout (LBM-Scenario-ImplicitIK). Single source of truth for
# the slicing used by from_sim_flat / to_sim_flat.
SIM_ACTION_DIM = 16
_L_XYZ = slice(0, 3)
_L_QUAT = slice(3, 7)
_R_XYZ = slice(7, 10)
_R_QUAT = slice(10, 14)
_L_GRIP = 14
_R_GRIP = 15

# robot arm key -> attribute name on BimanualAction.
ARM_KEYS: tuple[str, ...] = ("l", "r")


def _as_vec(a, n: int, name: str) -> np.ndarray:
    v = np.asarray(a, dtype=np.float64).reshape(-1)
    if v.shape[0] != n:
        raise ValueError(f"{name} must have {n} elements, got {v.shape[0]}")
    return v


@dataclass
class Pose:
    """Absolute EE pose in the sim env frame. Quaternion stored WXYZ."""

    pos: np.ndarray        # (3,) xyz [m]
    quat_wxyz: np.ndarray  # (4,) orientation, wxyz

    def __post_init__(self) -> None:
        self.pos = _as_vec(self.pos, 3, "pos")
        self.quat_wxyz = _as_vec(self.quat_wxyz, 4, "quat_wxyz")

    @property
    def quat_xyzw(self) -> np.ndarray:
        """Orientation as XYZW (scipy / franky convention)."""
        w, x, y, z = self.quat_wxyz
        return np.array([x, y, z, w], dtype=np.float64)

    @classmethod
    def from_quat_xyzw(cls, pos, quat_xyzw) -> "Pose":
        x, y, z, w = _as_vec(quat_xyzw, 4, "quat_xyzw")
        return cls(pos=pos, quat_wxyz=np.array([w, x, y, z], dtype=np.float64))


@dataclass
class ArmAction:
    """One arm's LBM action slice: absolute env-frame pose + gripper command."""

    pose: Pose
    gripper: float = 0.0

    def __post_init__(self) -> None:
        self.gripper = float(self.gripper)


@dataclass
class BimanualAction:
    """LBM-Scenario-ImplicitIK action: per-arm absolute pose + gripper.

    Either arm may be ``None`` to command a single arm (the robot drives only
    arms that are both active and present here).
    """

    left: ArmAction | None = None
    right: ArmAction | None = None

    def arm(self, key: str) -> ArmAction | None:
        """The ArmAction for robot key 'l'/'r' (None if unset)."""
        return {"l": self.left, "r": self.right}[key]

    # -- sim flat 16-vector (the LBM ImplicitIK action a policy emits) ---------
    @classmethod
    def from_sim_flat(cls, vec) -> "BimanualAction":
        v = _as_vec(vec, SIM_ACTION_DIM, "sim action vector")
        return cls(
            left=ArmAction(Pose(v[_L_XYZ], v[_L_QUAT]), float(v[_L_GRIP])),
            right=ArmAction(Pose(v[_R_XYZ], v[_R_QUAT]), float(v[_R_GRIP])),
        )

    def to_sim_flat(self) -> np.ndarray:
        if self.left is None or self.right is None:
            raise ValueError("to_sim_flat requires both arms set")
        v = np.zeros(SIM_ACTION_DIM, dtype=np.float64)
        v[_L_XYZ], v[_L_QUAT] = self.left.pose.pos, self.left.pose.quat_wxyz
        v[_R_XYZ], v[_R_QUAT] = self.right.pose.pos, self.right.pose.quat_wxyz
        v[_L_GRIP], v[_R_GRIP] = self.left.gripper, self.right.gripper
        return v

    # -- LeRobot dict ({arm}_{x,y,z,qx,qy,qz,qw,gripper}; quats XYZW) ----------
    @classmethod
    def from_robot_action(cls, action: dict) -> "BimanualAction":
        def parse(prefix: str) -> ArmAction | None:
            pose_keys = [f"{prefix}_{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            if not all(k in action for k in pose_keys):
                return None
            pos = [action[f"{prefix}_{k}"] for k in ("x", "y", "z")]
            quat_xyzw = [action[f"{prefix}_{k}"] for k in ("qx", "qy", "qz", "qw")]
            grip = float(action.get(f"{prefix}_gripper", 0.0))
            return ArmAction(Pose.from_quat_xyzw(pos, quat_xyzw), grip)

        return cls(left=parse("l"), right=parse("r"))

    def to_robot_action(self) -> dict:
        out: dict[str, float] = {}
        for key in ARM_KEYS:
            a = self.arm(key)
            if a is None:
                continue
            for axis, val in zip(("x", "y", "z"), a.pose.pos):
                out[f"{key}_{axis}"] = float(val)
            for axis, val in zip(("qx", "qy", "qz", "qw"), a.pose.quat_xyzw):
                out[f"{key}_{axis}"] = float(val)
            out[f"{key}_gripper"] = float(a.gripper)
        return out


__all__ = ["Pose", "ArmAction", "BimanualAction", "SIM_ACTION_DIM", "ARM_KEYS"]
