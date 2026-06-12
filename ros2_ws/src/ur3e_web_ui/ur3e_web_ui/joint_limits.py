from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

UR3E_JOINT_LIMITS_PATH = Path("/opt/ros/humble/share/ur_description/config/ur3e/joint_limits.yaml")


@dataclass(frozen=True)
class JointLimit:
    min_position: float
    max_position: float
    max_velocity: float

    @property
    def has_position_limits(self) -> bool:
        return math.isfinite(self.min_position) and math.isfinite(self.max_position)


class _DegreesLoader(yaml.SafeLoader):
    """SafeLoader that understands the !degrees tag used by ur_description."""


_DegreesLoader.add_constructor(
    "!degrees",
    lambda loader, node: math.radians(float(loader.construct_scalar(node))),
)


def load_ur3e_joint_limits(path: Path | str = UR3E_JOINT_LIMITS_PATH) -> dict[str, JointLimit]:
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.load(file, Loader=_DegreesLoader)

    limits: dict[str, JointLimit] = {}
    for name, entry in data["joint_limits"].items():
        if entry.get("has_position_limits", False):
            min_position = float(entry["min_position"])
            max_position = float(entry["max_position"])
        else:
            min_position = -math.inf
            max_position = math.inf
        limits[name] = JointLimit(
            min_position=min_position,
            max_position=max_position,
            max_velocity=float(entry["max_velocity"]) if entry.get("has_velocity_limits", False) else math.inf,
        )
    return limits


def clamp_to_limits(joint_name: str, value: float, limits: dict[str, JointLimit]) -> tuple[float, bool]:
    """Clamp value into the joint's position range. Returns (clamped_value, was_clamped)."""
    limit = limits.get(joint_name)
    if limit is None or not limit.has_position_limits:
        return value, False
    clamped = min(max(value, limit.min_position), limit.max_position)
    return clamped, clamped != value
