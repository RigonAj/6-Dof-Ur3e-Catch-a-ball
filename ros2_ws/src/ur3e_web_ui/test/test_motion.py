import math

import pytest

from ur3e_rollout_replay.replay_core import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_ROLLOUT_PATH,
    ReplayDataError,
    SafetyLimits,
    build_replay_plan,
    load_episode_targets,
)

from ur3e_web_ui import motion
from ur3e_web_ui.joint_limits import load_ur3e_joint_limits

LIMITS = load_ur3e_joint_limits()
SAFETY = SafetyLimits()
ZERO = (0.0,) * 6


def test_jog_basic():
    plan, target, clamped = motion.build_jog_target(ZERO, "elbow_joint", 1, 0.05, LIMITS)
    assert not clamped
    assert math.isclose(target, 0.05)
    assert len(plan.positions) == 1
    assert math.isclose(plan.positions[0][2], 0.05)
    # other joints unchanged
    assert plan.positions[0][0] == 0.0
    assert plan.time_from_start_s[0] == motion.JOG_MIN_DURATION


def test_jog_clamps_at_limit():
    near_limit = (0.0, 0.0, math.pi - 0.01, 0.0, 0.0, 0.0)
    plan, target, clamped = motion.build_jog_target(near_limit, "elbow_joint", 1, 0.05, LIMITS)
    assert clamped
    assert math.isclose(target, math.pi)
    # duration uses the actual (smaller) step but keeps the floor
    assert plan.time_from_start_s[0] == motion.JOG_MIN_DURATION


def test_jog_rejects_bad_input():
    with pytest.raises(ReplayDataError):
        motion.build_jog_target(ZERO, "not_a_joint", 1, 0.05, LIMITS)
    with pytest.raises(ReplayDataError):
        motion.build_jog_target(ZERO, "elbow_joint", 2, 0.05, LIMITS)
    with pytest.raises(ReplayDataError):
        motion.build_jog_target(ZERO, "elbow_joint", 1, motion.JOG_STEP_MAX + 0.01, LIMITS)


def test_home_plan_duration_floor():
    near_home = tuple(value + 0.01 for value in motion.HOME_POSITIONS)
    plan = motion.build_home_plan(near_home, SAFETY)
    assert plan.positions == (motion.HOME_POSITIONS,)
    assert plan.time_from_start_s[0] >= motion.HOME_MIN_DURATION


def test_wrap_joints_toward_seed_removes_full_turns():
    seed = (-1.354, 0.199, 0.307, -1.998, -0.311, 0.0)
    # Solution equivalent to the seed but offset by full turns on three joints.
    solution = (
        seed[0] + math.tau,
        seed[1] - math.tau,
        seed[2],
        seed[3] + math.tau,
        seed[4],
        seed[5],
    )
    wrapped = motion.wrap_joints_toward_seed(solution, seed, LIMITS)
    assert wrapped == pytest.approx(seed)


def test_wrap_joints_respects_position_limits():
    # elbow_joint is limited to +/-pi; a wrap candidate outside must be skipped.
    seed = (0.0, 0.0, 3.0, 0.0, 0.0, 0.0)
    solution = (0.0, 0.0, -3.0, 0.0, 0.0, 0.0)
    wrapped = motion.wrap_joints_toward_seed(solution, seed, LIMITS)
    # -3.0 + tau = 3.28 > pi, outside the elbow limit, so the value is kept.
    assert wrapped[2] == pytest.approx(-3.0)


def test_select_closest_ik_solution_prefers_near_branch():
    seed = (-1.354, 0.199, 0.307, -1.998, -0.311, 0.0)
    near = (-1.446, -0.479, 1.307, -2.049, -0.325, -0.284)
    far = (2.309, -4.157, 1.875, 4.842, 2.564, 1.151)
    assert motion.select_closest_ik_solution([far, near], seed) == pytest.approx(near)
    assert motion.select_closest_ik_solution([near, far], seed) == pytest.approx(near)


def test_max_joint_delta():
    assert motion.max_joint_delta((0.0,) * 6, (0.0, 0.1, -0.4, 0.0, 0.2, 0.0)) == pytest.approx(0.4)


def test_episode_plan_matches_replay_core():
    plan, episode, within = motion.build_episode_plan(
        DEFAULT_ROLLOUT_PATH, 0, SAFETY, current_positions=ZERO
    )
    reference_episode = load_episode_targets(DEFAULT_ROLLOUT_PATH, 0, DEFAULT_JOINT_NAMES)
    reference_plan = build_replay_plan(reference_episode, SAFETY, current_positions=ZERO)
    assert plan == reference_plan
    assert within


def test_episode_summaries():
    metadata, summaries = motion.episode_summaries(DEFAULT_ROLLOUT_PATH, SAFETY)
    assert metadata.get("dt_s") == pytest.approx(1.0 / 60.0)
    assert len(summaries) == 10
    for entry in summaries:
        assert entry["valid"]
        assert entry["steps"] > 0
        assert entry["retimed_total_s"] > 0


def test_plan_to_dict_round_trip():
    plan, _, _ = motion.build_episode_plan(DEFAULT_ROLLOUT_PATH, 0, SAFETY, current_positions=ZERO)
    payload = motion.plan_to_dict(plan, episode_index=0, within_limits=True)
    assert payload["episode_index"] == 0
    assert len(payload["positions"]) == len(payload["time_from_start_s"])
    assert payload["time_from_start_s"][0] == 0.0
    assert payload["retimed_stats"]["total_duration_s"] > 0
