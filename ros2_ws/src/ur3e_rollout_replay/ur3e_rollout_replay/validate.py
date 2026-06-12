from __future__ import annotations

import argparse
import sys

from .replay_core import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_ROLLOUT_PATH,
    ReplayDataError,
    SafetyLimits,
    build_replay_plan,
    load_episode_targets,
    plan_is_within_limits,
    raw_motion_is_within_limits,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an Isaac rollout before UR3e trajectory replay.")
    parser.add_argument("--rollout", default=str(DEFAULT_ROLLOUT_PATH), help="Path to rollouts_*.json")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to validate")
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
        "--fail-on-raw-unsafe",
        action="store_true",
        help="Return nonzero if the rollout is unsafe at metadata dt_s before retiming.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    limits = SafetyLimits(
        max_joint_velocity=args.max_joint_velocity,
        max_joint_acceleration=args.max_joint_acceleration,
        approach_min_duration=args.approach_min_duration,
        min_segment_duration=args.min_segment_duration,
    )

    try:
        episode = load_episode_targets(args.rollout, args.episode, DEFAULT_JOINT_NAMES)
        plan = build_replay_plan(episode, limits)
        raw_safe = raw_motion_is_within_limits(episode, limits)
        retimed_safe = plan_is_within_limits(plan, limits)
    except ReplayDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"rollout: {episode.rollout_path}")
    print(f"episode: {episode.episode_index}")
    print(f"samples: {len(episode.positions)}")
    print(f"metadata_dt_s: {episode.raw_dt_s:.9f}")
    print(f"joint_names: {', '.join(episode.joint_names)}")
    print("raw:")
    print(f"  max_step_rad: {plan.raw_stats.max_step_rad:.6f}")
    print(f"  max_velocity_rad_s: {plan.raw_stats.max_velocity_rad_s:.6f}")
    print(f"  max_acceleration_rad_s2: {plan.raw_stats.max_acceleration_rad_s2:.6f}")
    print(f"  total_duration_s: {plan.raw_stats.total_duration_s:.6f}")
    print(f"  safe_at_limits: {str(raw_safe).lower()}")
    print("retimed:")
    print(f"  max_velocity_rad_s: {plan.retimed_stats.max_velocity_rad_s:.6f}")
    print(f"  max_acceleration_rad_s2: {plan.retimed_stats.max_acceleration_rad_s2:.6f}")
    print(f"  total_duration_s: {plan.retimed_stats.total_duration_s:.6f}")
    print(f"  safe_at_limits: {str(retimed_safe).lower()}")
    print("limits:")
    print(f"  max_joint_velocity_rad_s: {limits.max_joint_velocity:.6f}")
    print(f"  max_joint_acceleration_rad_s2: {limits.max_joint_acceleration:.6f}")
    print(f"  min_segment_duration_s: {limits.min_segment_duration:.6f}")

    if args.fail_on_raw_unsafe and not raw_safe:
        print("ERROR: raw rollout timing exceeds requested limits", file=sys.stderr)
        return 1
    if not retimed_safe:
        print("ERROR: retimed rollout still exceeds requested limits", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
