from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_ROLLOUT_PATH = Path(
    "logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/rollouts_10_episodes.json"
)

DEFAULT_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class ReplayDataError(ValueError):
    """Raised when rollout data cannot safely be converted to joint targets."""


@dataclass(frozen=True)
class SafetyLimits:
    max_joint_velocity: float = 0.25
    max_joint_acceleration: float = 0.5
    approach_min_duration: float = 10.0
    min_segment_duration: float = 0.5

    def validate(self) -> None:
        if self.max_joint_velocity <= 0.0:
            raise ReplayDataError("--max-joint-velocity must be positive")
        if self.max_joint_acceleration <= 0.0:
            raise ReplayDataError("--max-joint-acceleration must be positive")
        if self.approach_min_duration < 0.0:
            raise ReplayDataError("--approach-min-duration must be nonnegative")
        if self.min_segment_duration <= 0.0:
            raise ReplayDataError("--min-segment-duration must be positive")


@dataclass(frozen=True)
class SegmentStats:
    max_step_rad: float
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float
    total_duration_s: float

    @property
    def safe_at(self) -> bool:
        return math.isfinite(self.max_velocity_rad_s) and math.isfinite(self.max_acceleration_rad_s2)


@dataclass(frozen=True)
class EpisodeTargets:
    rollout_path: Path
    metadata: dict[str, Any]
    episode_index: int
    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]

    @property
    def raw_dt_s(self) -> float:
        dt = self.metadata.get("dt_s")
        if not isinstance(dt, (int, float)) or not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ReplayDataError("rollout metadata.dt_s must be a positive finite number")
        return float(dt)


@dataclass(frozen=True)
class ReplayPlan:
    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]
    durations_s: tuple[float, ...]
    time_from_start_s: tuple[float, ...]
    raw_stats: SegmentStats
    retimed_stats: SegmentStats


def load_rollout(path: Path | str = DEFAULT_ROLLOUT_PATH) -> dict[str, Any]:
    resolved = resolve_rollout_path(path)
    try:
        with resolved.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ReplayDataError(f"rollout file does not exist: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayDataError(f"rollout file is not valid JSON: {resolved}: {exc}") from exc

    if not isinstance(data, dict):
        raise ReplayDataError("rollout JSON root must be an object")
    return data


def resolve_rollout_path(path: Path | str = DEFAULT_ROLLOUT_PATH) -> Path:
    requested = Path(path).expanduser()
    if requested.is_absolute():
        return requested

    search_roots = [Path.cwd(), *Path.cwd().parents]
    module_path = Path(__file__).resolve()
    search_roots.extend([module_path.parent, *module_path.parents])

    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / requested
        if candidate.exists():
            return candidate
    return requested


def load_episode_targets(
    path: Path | str = DEFAULT_ROLLOUT_PATH,
    episode_index: int = 0,
    expected_joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
) -> EpisodeTargets:
    resolved_path = resolve_rollout_path(path)
    data = load_rollout(resolved_path)
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ReplayDataError("rollout metadata must be an object")

    expected = tuple(expected_joint_names)
    metadata_joint_names = metadata.get("joint_names")
    if metadata_joint_names is not None and tuple(metadata_joint_names) != expected:
        raise ReplayDataError(
            "rollout metadata joint order does not match UR3e replay order: "
            f"expected {list(expected)}, got {metadata_joint_names}"
        )

    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ReplayDataError("rollout must contain a non-empty episodes list")
    if episode_index < 0 or episode_index >= len(episodes):
        raise ReplayDataError(f"episode index {episode_index} outside available range 0..{len(episodes) - 1}")

    episode = episodes[episode_index]
    if not isinstance(episode, dict):
        raise ReplayDataError(f"episode {episode_index} must be an object")
    samples = episode.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ReplayDataError(f"episode {episode_index} must contain a non-empty samples list")

    positions: list[tuple[float, ...]] = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ReplayDataError(f"sample {sample_index} in episode {episode_index} must be an object")
        if "joint_position_target_rad" not in sample:
            raise ReplayDataError(
                f"sample {sample_index} in episode {episode_index} is missing joint_position_target_rad"
            )
        positions.append(_coerce_joint_target(sample["joint_position_target_rad"], sample_index, expected))

    return EpisodeTargets(
        rollout_path=resolved_path,
        metadata=metadata,
        episode_index=episode_index,
        joint_names=expected,
        positions=tuple(positions),
    )


def _coerce_joint_target(value: Any, sample_index: int, expected_joint_names: Sequence[str]) -> tuple[float, ...]:
    if not isinstance(value, list | tuple):
        raise ReplayDataError(f"sample {sample_index} joint_position_target_rad must be a list")
    if len(value) != len(expected_joint_names):
        raise ReplayDataError(
            f"sample {sample_index} joint_position_target_rad must contain "
            f"{len(expected_joint_names)} values, got {len(value)}"
        )

    target: list[float] = []
    for joint_name, raw in zip(expected_joint_names, value, strict=True):
        if not isinstance(raw, (int, float)):
            raise ReplayDataError(f"sample {sample_index} target for {joint_name} is not numeric: {raw!r}")
        joint_value = float(raw)
        if not math.isfinite(joint_value):
            raise ReplayDataError(f"sample {sample_index} target for {joint_name} is not finite: {raw!r}")
        target.append(joint_value)
    return tuple(target)


def retime_segments(
    positions: Sequence[Sequence[float]],
    limits: SafetyLimits,
    min_duration: float | None = None,
) -> tuple[float, ...]:
    limits.validate()
    if len(positions) < 2:
        return tuple()

    floor = limits.min_segment_duration if min_duration is None else min_duration
    if floor < 0.0:
        raise ReplayDataError("segment duration floor must be nonnegative")

    durations = []
    for start, end in pairwise_positions(positions):
        max_delta = max(abs(b - a) for a, b in zip(start, end, strict=True))
        velocity_duration = max_delta / limits.max_joint_velocity if max_delta > 0.0 else 0.0
        acceleration_duration = math.sqrt(2.0 * max_delta / limits.max_joint_acceleration) if max_delta > 0.0 else 0.0
        durations.append(max(floor, velocity_duration, acceleration_duration))
    return tuple(durations)


def build_replay_plan(
    episode: EpisodeTargets,
    limits: SafetyLimits,
    current_positions: Sequence[float] | None = None,
) -> ReplayPlan:
    limits.validate()
    saved_positions = tuple(tuple(position) for position in episode.positions)
    raw_durations = tuple([episode.raw_dt_s] * max(0, len(saved_positions) - 1))
    raw_stats = compute_segment_stats(saved_positions, raw_durations)
    retimed_motion_durations = retime_segments(saved_positions, limits)

    if current_positions is None:
        positions = saved_positions
        durations = retimed_motion_durations
    else:
        current = _coerce_current_positions(current_positions, episode.joint_names)
        approach_duration = retime_segments(
            (current, saved_positions[0]),
            limits,
            min_duration=limits.approach_min_duration,
        )
        positions = (current, *saved_positions)
        durations = (*approach_duration, *retimed_motion_durations)

    retimed_stats = compute_segment_stats(positions, durations)
    return ReplayPlan(
        joint_names=episode.joint_names,
        positions=positions,
        durations_s=durations,
        time_from_start_s=durations_to_times(durations),
        raw_stats=raw_stats,
        retimed_stats=retimed_stats,
    )


def _coerce_current_positions(current_positions: Sequence[float], joint_names: Sequence[str]) -> tuple[float, ...]:
    if len(current_positions) != len(joint_names):
        raise ReplayDataError(f"current joint state must have {len(joint_names)} values, got {len(current_positions)}")
    current = tuple(float(value) for value in current_positions)
    for joint_name, value in zip(joint_names, current, strict=True):
        if not math.isfinite(value):
            raise ReplayDataError(f"current joint state for {joint_name} is not finite: {value!r}")
    return current


def durations_to_times(durations: Sequence[float]) -> tuple[float, ...]:
    times = [0.0]
    elapsed = 0.0
    for duration in durations:
        elapsed += duration
        times.append(elapsed)
    return tuple(times)


def compute_segment_stats(positions: Sequence[Sequence[float]], durations_s: Sequence[float]) -> SegmentStats:
    if len(positions) < 2:
        return SegmentStats(0.0, 0.0, 0.0, 0.0)
    if len(durations_s) != len(positions) - 1:
        raise ReplayDataError(
            f"expected {len(positions) - 1} segment durations for {len(positions)} positions, got {len(durations_s)}"
        )

    max_step = 0.0
    max_velocity = 0.0
    max_acceleration = 0.0
    total_duration = 0.0
    for duration, (start, end) in zip(durations_s, pairwise_positions(positions), strict=True):
        if not math.isfinite(duration) or duration <= 0.0:
            raise ReplayDataError(f"segment duration must be positive and finite, got {duration!r}")
        total_duration += duration
        for a, b in zip(start, end, strict=True):
            delta = abs(b - a)
            max_step = max(max_step, delta)
            max_velocity = max(max_velocity, delta / duration)
            max_acceleration = max(max_acceleration, 2.0 * delta / (duration * duration))

    return SegmentStats(max_step, max_velocity, max_acceleration, total_duration)


def pairwise_positions(positions: Sequence[Sequence[float]]) -> Iterable[tuple[Sequence[float], Sequence[float]]]:
    return zip(positions, positions[1:], strict=False)


def plan_is_within_limits(plan: ReplayPlan, limits: SafetyLimits) -> bool:
    tolerance = 1.0e-9
    return (
        plan.retimed_stats.max_velocity_rad_s <= limits.max_joint_velocity + tolerance
        and plan.retimed_stats.max_acceleration_rad_s2 <= limits.max_joint_acceleration + tolerance
    )


def raw_motion_is_within_limits(episode: EpisodeTargets, limits: SafetyLimits) -> bool:
    raw_durations = tuple([episode.raw_dt_s] * max(0, len(episode.positions) - 1))
    raw_stats = compute_segment_stats(episode.positions, raw_durations)
    tolerance = 1.0e-9
    return (
        raw_stats.max_velocity_rad_s <= limits.max_joint_velocity + tolerance
        and raw_stats.max_acceleration_rad_s2 <= limits.max_joint_acceleration + tolerance
    )
