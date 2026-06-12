from __future__ import annotations

import argparse
import sys
from typing import Any

from .replay_core import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_ROLLOUT_PATH,
    ReplayDataError,
    SafetyLimits,
    build_replay_plan,
    load_episode_targets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a retimed Isaac rollout to a UR joint trajectory controller.")
    parser.add_argument("--rollout", default=str(DEFAULT_ROLLOUT_PATH), help="Path to rollouts_*.json")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument(
        "--action-name",
        default="/scaled_joint_trajectory_controller/follow_joint_trajectory",
        help="FollowJointTrajectory action name",
    )
    parser.add_argument(
        "--joint-state-topic", default="/joint_states", help="JointState topic to read current pose from"
    )
    parser.add_argument(
        "--joint-state-timeout", type=float, default=10.0, help="Seconds to wait for current joint state"
    )
    parser.add_argument("--max-joint-velocity", type=float, default=0.25, help="Safety velocity limit in rad/s")
    parser.add_argument(
        "--max-joint-acceleration", type=float, default=0.5, help="Safety acceleration limit in rad/s^2"
    )
    parser.add_argument(
        "--approach-min-duration", type=float, default=10.0, help="Minimum approach duration in seconds"
    )
    parser.add_argument(
        "--min-segment-duration", type=float, default=0.5, help="Minimum replay segment duration in seconds"
    )
    parser.add_argument(
        "--current-joints",
        help="Comma-separated six-joint current pose override in replay joint order; skips /joint_states wait",
    )
    parser.add_argument("--execute", action="store_true", help="Actually send the trajectory goal")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from control_msgs.action import FollowJointTrajectory
        from builtin_interfaces.msg import Duration
    except ImportError as exc:
        print(
            "ERROR: ROS 2 action dependencies are missing. Install/build control_msgs and source the ROS workspace "
            "before running ur3e_replay_send.",
            file=sys.stderr,
        )
        print(f"missing import: {exc}", file=sys.stderr)
        return 2

    args = build_parser().parse_args(argv)
    limits = SafetyLimits(
        max_joint_velocity=args.max_joint_velocity,
        max_joint_acceleration=args.max_joint_acceleration,
        approach_min_duration=args.approach_min_duration,
        min_segment_duration=args.min_segment_duration,
    )

    try:
        episode = load_episode_targets(args.rollout, args.episode, DEFAULT_JOINT_NAMES)
        current_override = parse_current_joints(args.current_joints) if args.current_joints else None
    except ReplayDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    class ReplaySender(Node):
        def __init__(self) -> None:
            super().__init__("ur3e_rollout_replay_sender")
            self._joint_state: dict[str, float] | None = None
            self._subscription = self.create_subscription(JointState, args.joint_state_topic, self._on_joint_state, 10)
            self._client = ActionClient(self, FollowJointTrajectory, args.action_name)

        def _on_joint_state(self, msg: Any) -> None:
            by_name = dict(zip(msg.name, msg.position, strict=False))
            if all(name in by_name for name in DEFAULT_JOINT_NAMES):
                self._joint_state = {name: float(by_name[name]) for name in DEFAULT_JOINT_NAMES}

        def current_positions(self) -> tuple[float, ...] | None:
            if self._joint_state is None:
                return None
            return tuple(self._joint_state[name] for name in DEFAULT_JOINT_NAMES)

    rclpy.init(args=None)
    node = ReplaySender()
    try:
        if current_override is None:
            deadline = node.get_clock().now().nanoseconds + int(args.joint_state_timeout * 1_000_000_000)
            while rclpy.ok() and node.current_positions() is None and node.get_clock().now().nanoseconds < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            current_positions = node.current_positions()
            if current_positions is None:
                print(
                    f"ERROR: no complete UR3e joint state received on {args.joint_state_topic} "
                    f"within {args.joint_state_timeout:.1f}s",
                    file=sys.stderr,
                )
                return 2
        else:
            current_positions = current_override

        plan = build_replay_plan(episode, limits, current_positions=current_positions)
        print_plan_summary(plan, execute=args.execute)
        if not args.execute:
            print("dry_run: true")
            print("pass --execute to send this trajectory")
            return 0

        if not node._client.wait_for_server(timeout_sec=5.0):
            print(f"ERROR: action server not available: {args.action_name}", file=sys.stderr)
            return 2

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = build_joint_trajectory(plan, JointTrajectory, JointTrajectoryPoint, Duration)
        future = node._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            print("ERROR: trajectory goal was rejected", file=sys.stderr)
            return 1

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result()
        if result is None:
            print("ERROR: no trajectory result returned", file=sys.stderr)
            return 1
        error_code = getattr(result.result, "error_code", 0)
        error_string = getattr(result.result, "error_string", "")
        print(f"result_error_code: {error_code}")
        if error_string:
            print(f"result_error_string: {error_string}")
        return 0 if error_code == 0 else 1
    except ReplayDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_current_joints(value: str) -> tuple[float, ...]:
    try:
        joints = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ReplayDataError("--current-joints must be a comma-separated list of floats") from exc
    if len(joints) != len(DEFAULT_JOINT_NAMES):
        raise ReplayDataError(f"--current-joints must contain {len(DEFAULT_JOINT_NAMES)} values")
    return joints


def seconds_to_duration(seconds: float, duration_type: type) -> Any:
    whole = int(seconds)
    nanoseconds = int(round((seconds - whole) * 1_000_000_000))
    if nanoseconds >= 1_000_000_000:
        whole += 1
        nanoseconds -= 1_000_000_000
    msg = duration_type()
    msg.sec = whole
    msg.nanosec = nanoseconds
    return msg


def build_joint_trajectory(plan: Any, trajectory_type: type, point_type: type, duration_type: type) -> Any:
    trajectory = trajectory_type()
    trajectory.joint_names = list(plan.joint_names)
    trajectory.points = []
    for positions, time_from_start in zip(plan.positions, plan.time_from_start_s, strict=True):
        point = point_type()
        point.positions = list(positions)
        point.time_from_start = seconds_to_duration(time_from_start, duration_type)
        trajectory.points.append(point)
    return trajectory


def print_plan_summary(plan: Any, execute: bool) -> None:
    print(f"execute: {str(execute).lower()}")
    print(f"points: {len(plan.positions)}")
    print(f"segments: {len(plan.durations_s)}")
    print(f"total_duration_s: {plan.time_from_start_s[-1]:.6f}")
    print(f"max_velocity_rad_s: {plan.retimed_stats.max_velocity_rad_s:.6f}")
    print(f"max_acceleration_rad_s2: {plan.retimed_stats.max_acceleration_rad_s2:.6f}")
    print(f"first_target_rad: {[round(v, 6) for v in plan.positions[1]] if len(plan.positions) > 1 else []}")
    print(f"final_target_rad: {[round(v, 6) for v in plan.positions[-1]]}")


if __name__ == "__main__":
    raise SystemExit(main())
