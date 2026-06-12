from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from ur3e_rollout_replay.replay_core import (
    DEFAULT_JOINT_NAMES,
    EpisodeTargets,
    ReplayDataError,
    ReplayPlan,
    SafetyLimits,
    build_replay_plan,
    compute_segment_stats,
    durations_to_times,
    load_episode_targets,
    load_rollout,
    plan_is_within_limits,
    raw_motion_is_within_limits,
    resolve_rollout_path,
    retime_segments,
)

from .joint_limits import JointLimit, clamp_to_limits

JOG_STEP_DEFAULT = 0.05
JOG_STEP_MAX = 0.2
JOG_VELOCITY = 0.15
JOG_MIN_DURATION = 0.4

HOME_POSITIONS = (0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0)
HOME_MIN_DURATION = 5.0
TCP_TARGET_MIN_DURATION = 2.0

# Penalize base-joint motion more than wrist motion when comparing IK branches:
# a shoulder flip sweeps the whole arm, a wrist twist barely moves the TCP path.
IK_DISTANCE_WEIGHTS = (3.0, 2.5, 2.0, 1.0, 1.0, 0.5)


@dataclass(frozen=True)
class SimplePlan:
    """Minimal plan compatible with ur3e_rollout_replay.send.build_joint_trajectory."""

    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]
    time_from_start_s: tuple[float, ...]


def build_jog_target(
    base_positions: Sequence[float],
    joint_name: str,
    direction: int,
    step_rad: float,
    limits: dict[str, JointLimit],
) -> tuple[SimplePlan, float, bool]:
    """Build a single-point trajectory moving one joint by +/- step.

    Returns (plan, target_rad, clamped).
    """
    if joint_name not in DEFAULT_JOINT_NAMES:
        raise ReplayDataError(f"unknown joint: {joint_name}")
    if direction not in (1, -1):
        raise ReplayDataError("direction must be 1 or -1")
    if not math.isfinite(step_rad) or step_rad <= 0.0 or step_rad > JOG_STEP_MAX:
        raise ReplayDataError(f"step_rad must be in (0, {JOG_STEP_MAX}]")
    if len(base_positions) != len(DEFAULT_JOINT_NAMES):
        raise ReplayDataError(f"base positions must have {len(DEFAULT_JOINT_NAMES)} values")

    index = DEFAULT_JOINT_NAMES.index(joint_name)
    target_value = float(base_positions[index]) + direction * step_rad
    target_value, clamped = clamp_to_limits(joint_name, target_value, limits)

    target = list(float(value) for value in base_positions)
    actual_step = abs(target_value - target[index])
    target[index] = target_value
    duration = max(JOG_MIN_DURATION, actual_step / JOG_VELOCITY)

    plan = SimplePlan(
        joint_names=DEFAULT_JOINT_NAMES,
        positions=(tuple(target),),
        time_from_start_s=(duration,),
    )
    return plan, target_value, clamped


def build_joint_target_plan(
    current_positions: Sequence[float],
    target_positions: Sequence[float],
    safety_limits: SafetyLimits,
    min_duration: float = TCP_TARGET_MIN_DURATION,
) -> SimplePlan:
    """Single-segment retimed move from the current joint pose to an IK target."""
    if len(current_positions) != len(DEFAULT_JOINT_NAMES):
        raise ReplayDataError(f"current positions must have {len(DEFAULT_JOINT_NAMES)} values")
    if len(target_positions) != len(DEFAULT_JOINT_NAMES):
        raise ReplayDataError(f"target positions must have {len(DEFAULT_JOINT_NAMES)} values")
    current = tuple(float(value) for value in current_positions)
    target = tuple(float(value) for value in target_positions)
    for joint_name, value in zip((*DEFAULT_JOINT_NAMES, *DEFAULT_JOINT_NAMES), (*current, *target), strict=True):
        if not math.isfinite(value):
            raise ReplayDataError(f"joint target for {joint_name} is not finite: {value!r}")
    durations = retime_segments((current, target), safety_limits, min_duration=min_duration)
    return SimplePlan(
        joint_names=DEFAULT_JOINT_NAMES,
        positions=(target,),
        time_from_start_s=durations_to_times(durations)[1:],
    )


def build_home_plan(
    current_positions: Sequence[float],
    safety_limits: SafetyLimits,
    home_positions: Sequence[float] = HOME_POSITIONS,
) -> SimplePlan:
    """Single-segment retimed move from current pose to home."""
    if len(current_positions) != len(DEFAULT_JOINT_NAMES):
        raise ReplayDataError(f"current positions must have {len(DEFAULT_JOINT_NAMES)} values")
    current = tuple(float(value) for value in current_positions)
    home = tuple(float(value) for value in home_positions)
    durations = retime_segments((current, home), safety_limits, min_duration=HOME_MIN_DURATION)
    return SimplePlan(
        joint_names=DEFAULT_JOINT_NAMES,
        positions=(home,),
        time_from_start_s=durations_to_times(durations)[1:],
    )


def wrap_joints_toward_seed(
    solution: Sequence[float],
    seed: Sequence[float],
    limits: dict[str, JointLimit],
) -> tuple[float, ...]:
    """Shift each joint by whole turns (2*pi) to land as close to the seed as possible.

    All UR3e joints are revolute with +/-2*pi limits, so adding a full turn reaches
    the identical TCP pose; this only removes needless full rotations from an IK
    solution. Candidates outside the joint position limits are skipped.
    """
    if len(solution) != len(DEFAULT_JOINT_NAMES) or len(seed) != len(DEFAULT_JOINT_NAMES):
        raise ReplayDataError(f"IK solution and seed must have {len(DEFAULT_JOINT_NAMES)} values")
    wrapped = []
    for joint_name, value, reference in zip(DEFAULT_JOINT_NAMES, solution, seed, strict=True):
        value = float(value)
        reference = float(reference)
        limit = limits.get(joint_name)
        best = value
        turns = round((reference - value) / math.tau)
        for whole_turns in (turns - 1, turns, turns + 1):
            candidate = value + whole_turns * math.tau
            if limit is not None and limit.has_position_limits:
                if candidate < limit.min_position or candidate > limit.max_position:
                    continue
            if abs(candidate - reference) < abs(best - reference):
                best = candidate
        wrapped.append(best)
    return tuple(wrapped)


def max_joint_delta(solution: Sequence[float], seed: Sequence[float]) -> float:
    """Largest single-joint distance between an IK solution and the seed pose."""
    return max(abs(float(a) - float(b)) for a, b in zip(solution, seed, strict=True))


def select_closest_ik_solution(
    solutions: Sequence[Sequence[float]],
    seed: Sequence[float],
) -> tuple[float, ...]:
    """Pick the IK branch needing the least (weighted) joint motion from the seed."""
    if not solutions:
        raise ReplayDataError("no IK solutions to select from")

    def weighted_distance(solution: Sequence[float]) -> float:
        return sum(
            weight * abs(float(a) - float(b))
            for weight, a, b in zip(IK_DISTANCE_WEIGHTS, solution, seed, strict=True)
        )

    return tuple(float(value) for value in min(solutions, key=weighted_distance))


def build_episode_plan(
    rollout_path: Path | str,
    episode_index: int,
    safety_limits: SafetyLimits,
    current_positions: Sequence[float] | None = None,
) -> tuple[ReplayPlan, EpisodeTargets, bool]:
    """Load one episode and build the retimed (optionally approach-prefixed) plan.

    Returns (plan, episode, within_limits).
    """
    episode = load_episode_targets(rollout_path, episode_index, DEFAULT_JOINT_NAMES)
    plan = build_replay_plan(episode, safety_limits, current_positions=current_positions)
    return plan, episode, plan_is_within_limits(plan, safety_limits)


_summaries_cache: dict[tuple[str, float, SafetyLimits], list[dict]] = {}


def episode_summaries(rollout_path: Path | str, safety_limits: SafetyLimits) -> tuple[dict, list[dict]]:
    """Returns (metadata, per-episode summaries). Cached on file mtime."""
    resolved = resolve_rollout_path(rollout_path)
    try:
        mtime = resolved.stat().st_mtime
    except OSError as exc:
        raise ReplayDataError(f"rollout file does not exist: {resolved}") from exc

    cache_key = (str(resolved), mtime, safety_limits)
    data = load_rollout(resolved)
    metadata = data.get("metadata") or {}

    if cache_key in _summaries_cache:
        return metadata, _summaries_cache[cache_key]

    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise ReplayDataError("rollout must contain an episodes list")

    summaries: list[dict] = []
    for index in range(len(episodes)):
        entry: dict = {"index": index}
        raw = episodes[index] if isinstance(episodes[index], dict) else {}
        entry["success"] = bool(raw.get("success", False))
        entry["steps"] = int(raw.get("steps", len(raw.get("samples", []) or [])))
        try:
            episode = load_episode_targets(resolved, index, DEFAULT_JOINT_NAMES)
            entry["raw_safe"] = raw_motion_is_within_limits(episode, safety_limits)
            durations = retime_segments(episode.positions, safety_limits)
            entry["retimed_total_s"] = round(sum(durations), 3)
            entry["valid"] = True
        except ReplayDataError as exc:
            entry["valid"] = False
            entry["error"] = str(exc)
        summaries.append(entry)

    _summaries_cache.clear()
    _summaries_cache[cache_key] = summaries
    return metadata, summaries


def plan_to_dict(plan: ReplayPlan | SimplePlan, **extra) -> dict:
    """JSON-friendly representation shared by /plan responses and previews."""
    payload: dict = {
        "joint_names": list(plan.joint_names),
        "positions": [list(point) for point in plan.positions],
        "time_from_start_s": list(plan.time_from_start_s),
    }
    if isinstance(plan, ReplayPlan):
        payload["durations_s"] = list(plan.durations_s)
        payload["raw_stats"] = _stats_to_dict(plan.raw_stats)
        payload["retimed_stats"] = _stats_to_dict(plan.retimed_stats)
    payload.update(extra)
    return payload


def _stats_to_dict(stats) -> dict:
    return {
        "max_step_rad": stats.max_step_rad,
        "max_velocity_rad_s": stats.max_velocity_rad_s,
        "max_acceleration_rad_s2": stats.max_acceleration_rad_s2,
        "total_duration_s": stats.total_duration_s,
    }
