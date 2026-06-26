"""Configuration for the bimanual SpaceMouse teleoperator.

Uses ``SpaceMouseLeaderFields`` (plain dataclass) rather than
``SpaceMouseConfig`` (TeleoperatorConfig subclass) to avoid draccus recursing
through the choice registry when building the CLI parser.

The two SpaceMice are expected on different hidraw nodes (e.g. /dev/hidraw4
for the left arm and /dev/hidraw5 for the right arm).
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_spacemouse import SpaceMouseLeaderFields


def _left_defaults() -> SpaceMouseLeaderFields:
    return SpaceMouseLeaderFields(hidraw_path="/dev/hidraw2")


def _right_defaults() -> SpaceMouseLeaderFields:
    return SpaceMouseLeaderFields(hidraw_path="/dev/hidraw3")


@TeleoperatorConfig.register_subclass("bimanual_spacemouse")
@dataclass
class BimanualSpaceMouseConfig(TeleoperatorConfig):
    """Pair of SpaceMouse leaders driving a bimanual follower (left + right arm)."""

    left_arm_config: SpaceMouseLeaderFields = field(default_factory=_left_defaults)
    right_arm_config: SpaceMouseLeaderFields = field(default_factory=_right_defaults)
