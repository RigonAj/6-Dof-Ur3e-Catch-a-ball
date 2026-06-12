from __future__ import annotations

from dataclasses import dataclass, field

from ur3e_rollout_replay.replay_core import DEFAULT_JOINT_NAMES, SafetyLimits, build_replay_plan, load_episode_targets
from ur3e_rollout_replay.send import build_joint_trajectory


class FakeDuration:
    def __init__(self) -> None:
        self.sec = 0
        self.nanosec = 0


class FakeJointTrajectoryPoint:
    def __init__(self) -> None:
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.time_from_start = FakeDuration()


@dataclass
class FakeJointTrajectory:
    joint_names: list[str] = field(default_factory=list)
    points: list[FakeJointTrajectoryPoint] = field(default_factory=list)


def test_trajectory_builder_preserves_joint_order_and_leaves_velocity_fields_empty() -> None:
    episode = load_episode_targets(episode_index=0)
    plan = build_replay_plan(
        episode,
        SafetyLimits(),
        current_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    trajectory = build_joint_trajectory(plan, FakeJointTrajectory, FakeJointTrajectoryPoint, FakeDuration)

    assert trajectory.joint_names == list(DEFAULT_JOINT_NAMES)
    assert len(trajectory.points) == len(plan.positions)
    assert trajectory.points[0].positions == [0.0] * 6
    assert trajectory.points[1].positions == list(episode.positions[0])
    assert trajectory.points[0].velocities == []
    assert trajectory.points[0].accelerations == []
    assert trajectory.points[-1].time_from_start.sec >= 1
