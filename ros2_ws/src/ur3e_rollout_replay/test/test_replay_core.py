from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ur3e_rollout_replay.replay_core import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_ROLLOUT_PATH,
    ReplayDataError,
    SafetyLimits,
    build_replay_plan,
    load_episode_targets,
    plan_is_within_limits,
    raw_motion_is_within_limits,
)
from ur3e_rollout_replay.validate import main as validate_main


def test_default_rollout_is_raw_unsafe_but_retimed_safe() -> None:
    episode = load_episode_targets(DEFAULT_ROLLOUT_PATH, 0)
    limits = SafetyLimits()

    assert len(episode.positions) == 11
    assert not raw_motion_is_within_limits(episode, limits)

    plan = build_replay_plan(episode, limits)

    assert plan.raw_stats.max_velocity_rad_s > 100.0
    assert plan.raw_stats.max_acceleration_rad_s2 > 10_000.0
    assert plan_is_within_limits(plan, limits)
    assert plan.retimed_stats.max_velocity_rad_s <= limits.max_joint_velocity + 1.0e-9
    assert plan.retimed_stats.max_acceleration_rad_s2 <= limits.max_joint_acceleration + 1.0e-9
    assert plan.time_from_start_s[-1] > plan.raw_stats.total_duration_s


def test_approach_from_current_pose_is_inserted() -> None:
    episode = load_episode_targets(DEFAULT_ROLLOUT_PATH, 0)
    limits = SafetyLimits()
    current = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    plan = build_replay_plan(episode, limits, current_positions=current)

    assert plan.positions[0] == current
    assert plan.positions[1] == episode.positions[0]
    assert plan.durations_s[0] >= limits.approach_min_duration
    assert len(plan.positions) == len(episode.positions) + 1


def test_validate_cli_default_allows_retimed_safe_rollout() -> None:
    assert validate_main(["--rollout", str(DEFAULT_ROLLOUT_PATH), "--episode", "0"]) == 0


def test_validate_cli_can_fail_on_raw_unsafe_rollout() -> None:
    assert (
        validate_main(
            [
                "--rollout",
                str(DEFAULT_ROLLOUT_PATH),
                "--episode",
                "0",
                "--fail-on-raw-unsafe",
            ]
        )
        == 1
    )


def test_rejects_metadata_joint_name_mismatch(tmp_path: Path) -> None:
    path = write_rollout(tmp_path, joint_names=("wrong_joint", *DEFAULT_JOINT_NAMES[1:]))

    with pytest.raises(ReplayDataError, match="joint order"):
        load_episode_targets(path, 0)


def test_rejects_missing_joint_position_target(tmp_path: Path) -> None:
    path = write_rollout(tmp_path, samples=[{}])

    with pytest.raises(ReplayDataError, match="missing joint_position_target_rad"):
        load_episode_targets(path, 0)


def test_rejects_nonfinite_joint_target(tmp_path: Path) -> None:
    bad_target = [0.0] * len(DEFAULT_JOINT_NAMES)
    bad_target[2] = math.inf
    path = write_rollout(tmp_path, samples=[{"joint_position_target_rad": bad_target}])

    with pytest.raises(ReplayDataError, match="not finite"):
        load_episode_targets(path, 0)


def test_rejects_malformed_target_length(tmp_path: Path) -> None:
    path = write_rollout(tmp_path, samples=[{"joint_position_target_rad": [0.0, 1.0]}])

    with pytest.raises(ReplayDataError, match="must contain 6 values"):
        load_episode_targets(path, 0)


def test_rejects_episode_out_of_range(tmp_path: Path) -> None:
    path = write_rollout(tmp_path)

    with pytest.raises(ReplayDataError, match="outside available range"):
        load_episode_targets(path, 1)


def write_rollout(
    tmp_path: Path,
    joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES,
    samples: list[dict] | None = None,
) -> Path:
    if samples is None:
        samples = [{"joint_position_target_rad": [0.0] * len(DEFAULT_JOINT_NAMES)}]
    path = tmp_path / "rollout.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "dt_s": 1.0 / 60.0,
                    "joint_names": list(joint_names),
                },
                "episodes": [{"samples": samples}],
            }
        ),
        encoding="utf-8",
    )
    return path
