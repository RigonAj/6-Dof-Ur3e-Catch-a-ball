#!/usr/bin/env python3
"""Compare a simulated rollout reference against a real UR joint-state log."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
from pathlib import Path


DEFAULT_ROLLOUT = (
    "logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/rollouts_10_episodes.json"
)
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
TIME_COLUMNS = ("time_s", "time", "timestamp", "t")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sim post-action joint positions from rollouts_*.json with a real UR3e joint-state CSV."
        )
    )
    parser.add_argument("--rollout", default=DEFAULT_ROLLOUT, help="Path to the rollout JSON exported by play.py.")
    parser.add_argument("--real-log", required=True, help="CSV with one column per UR3e joint in radians.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index inside the rollout JSON.")
    parser.add_argument(
        "--real-start-time",
        type=float,
        default=None,
        help="Optional real-log timestamp that corresponds to sim time 0.0. Values before it are ignored.",
    )
    parser.add_argument(
        "--time-column",
        default=None,
        help="CSV time column. Defaults to the first available of: time_s, time, timestamp, t.",
    )
    parser.add_argument(
        "--max-time-error",
        type=float,
        default=0.05,
        help="Warn if nearest real sample is farther than this many seconds from the sim sample.",
    )
    parser.add_argument("--output", default=None, help="Optional path to write a JSON comparison report.")
    return parser.parse_args()


def _load_sim_reference(path: str, episode_index: int) -> tuple[dict, list[float], list[list[float]]]:
    with open(path, encoding="utf-8") as file:
        rollout = json.load(file)

    episodes = rollout.get("episodes", [])
    if episode_index < 0 or episode_index >= len(episodes):
        raise SystemExit(f"Episode {episode_index} is out of range. Rollout contains {len(episodes)} episodes.")

    metadata = rollout.get("metadata", {})
    samples = episodes[episode_index].get("samples", [])
    missing = [i for i, sample in enumerate(samples) if "joint_position_after_rad" not in sample]
    if missing:
        raise SystemExit(
            "This rollout does not contain sim post-action positions. "
            "Regenerate it with the updated scripts/skrl/play.py using --record_actions."
        )

    dt = float(metadata.get("dt_s", 0.0))
    times = [float(sample.get("sim_time_after_s", sample["time_s"] + dt)) for sample in samples]
    positions = [sample["joint_position_after_rad"] for sample in samples]
    return metadata, times, positions


def _detect_time_column(fieldnames: list[str], requested: str | None) -> str | None:
    if requested:
        if requested not in fieldnames:
            raise SystemExit(f"Requested time column {requested!r} was not found in real log.")
        return requested
    for name in TIME_COLUMNS:
        if name in fieldnames:
            return name
    return None


def _load_real_csv(
    path: str,
    time_column: str | None,
    real_start_time: float | None,
) -> tuple[list[float] | None, list[list[float]]]:
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise SystemExit("Real log CSV has no header row.")
        missing = [name for name in JOINT_NAMES if name not in reader.fieldnames]
        if missing:
            raise SystemExit(
                "Real log CSV must contain joint columns in radians. Missing: " + ", ".join(missing)
            )
        detected_time_column = _detect_time_column(reader.fieldnames, time_column)

        times = []
        positions = []
        for row in reader:
            if detected_time_column is None:
                sample_time = None
            else:
                sample_time = float(row[detected_time_column])
                if real_start_time is not None:
                    sample_time -= real_start_time
                    if sample_time < 0.0:
                        continue
            times.append(sample_time)
            positions.append([float(row[name]) for name in JOINT_NAMES])

    if not positions:
        raise SystemExit("Real log CSV contains no usable joint samples.")
    return (None if times and times[0] is None else times), positions


def _nearest_index(times: list[float], target: float) -> tuple[int, float]:
    index = bisect.bisect_left(times, target)
    candidates = []
    if index < len(times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda i: abs(times[i] - target))
    return best, abs(times[best] - target)


def _align_samples(
    sim_times: list[float],
    sim_positions: list[list[float]],
    real_times: list[float] | None,
    real_positions: list[list[float]],
) -> tuple[list[dict], list[float]]:
    aligned = []
    time_errors = []
    if real_times is None:
        count = min(len(sim_positions), len(real_positions))
        for index in range(count):
            aligned.append(
                {
                    "sim_index": index,
                    "real_index": index,
                    "sim_time_s": sim_times[index],
                    "real_time_s": None,
                    "sim_position_rad": sim_positions[index],
                    "real_position_rad": real_positions[index],
                }
            )
        return aligned, time_errors

    for sim_index, sim_time in enumerate(sim_times):
        real_index, time_error = _nearest_index(real_times, sim_time)
        time_errors.append(time_error)
        aligned.append(
            {
                "sim_index": sim_index,
                "real_index": real_index,
                "sim_time_s": sim_time,
                "real_time_s": real_times[real_index],
                "time_error_s": time_error,
                "sim_position_rad": sim_positions[sim_index],
                "real_position_rad": real_positions[real_index],
            }
        )
    return aligned, time_errors


def _compute_metrics(aligned: list[dict]) -> dict:
    if not aligned:
        raise SystemExit("No aligned samples to compare.")

    errors = []
    for sample in aligned:
        errors.append(
            [
                real - sim
                for real, sim in zip(sample["real_position_rad"], sample["sim_position_rad"], strict=True)
            ]
        )

    per_joint_rms = []
    per_joint_max_abs = []
    for joint_index in range(len(JOINT_NAMES)):
        joint_errors = [row[joint_index] for row in errors]
        per_joint_rms.append(math.sqrt(sum(error * error for error in joint_errors) / len(joint_errors)))
        per_joint_max_abs.append(max(abs(error) for error in joint_errors))

    flat_errors = [error for row in errors for error in row]
    final_error = errors[-1]
    return {
        "samples_compared": len(aligned),
        "joint_names": JOINT_NAMES,
        "rms_all_rad": math.sqrt(sum(error * error for error in flat_errors) / len(flat_errors)),
        "max_abs_rad": max(abs(error) for error in flat_errors),
        "per_joint_rms_rad": dict(zip(JOINT_NAMES, per_joint_rms, strict=True)),
        "per_joint_max_abs_rad": dict(zip(JOINT_NAMES, per_joint_max_abs, strict=True)),
        "final_error_rad": dict(zip(JOINT_NAMES, final_error, strict=True)),
        "final_abs_error_rad": dict(zip(JOINT_NAMES, [abs(error) for error in final_error], strict=True)),
        "final_abs_error_deg": dict(
            zip(JOINT_NAMES, [math.degrees(abs(error)) for error in final_error], strict=True)
        ),
    }


def _print_report(metrics: dict, time_errors: list[float], max_time_error: float) -> None:
    print("Sim/real joint-position comparison")
    print(f"samples_compared: {metrics['samples_compared']}")
    print(f"rms_all_rad: {metrics['rms_all_rad']:.6f}")
    print(f"max_abs_rad: {metrics['max_abs_rad']:.6f}")
    if time_errors:
        print(f"max_time_alignment_error_s: {max(time_errors):.6f}")
        if max(time_errors) > max_time_error:
            print(f"WARNING: time alignment exceeded --max-time-error={max_time_error}")
    print("\nFinal absolute error:")
    for name in JOINT_NAMES:
        rad = metrics["final_abs_error_rad"][name]
        deg = metrics["final_abs_error_deg"][name]
        print(f"  {name}: {rad:.6f} rad ({deg:.3f} deg)")
    print("\nPer-joint RMS:")
    for name in JOINT_NAMES:
        print(f"  {name}: {metrics['per_joint_rms_rad'][name]:.6f} rad")


def main() -> None:
    args = _parse_args()
    metadata, sim_times, sim_positions = _load_sim_reference(args.rollout, args.episode)
    real_times, real_positions = _load_real_csv(args.real_log, args.time_column, args.real_start_time)
    aligned, time_errors = _align_samples(sim_times, sim_positions, real_times, real_positions)
    metrics = _compute_metrics(aligned)
    report = {
        "rollout": os.path.abspath(args.rollout),
        "real_log": os.path.abspath(args.real_log),
        "episode": args.episode,
        "metadata": metadata,
        "metrics": metrics,
        "time_alignment": {
            "used_time_column": real_times is not None,
            "max_time_error_s": max(time_errors) if time_errors else None,
            "mean_time_error_s": sum(time_errors) / len(time_errors) if time_errors else None,
        },
        "aligned_samples": aligned,
    }
    _print_report(metrics, time_errors, args.max_time_error)

    if args.output:
        output_path = Path(args.output)
        if output_path.parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(f"\nWrote comparison report: {output_path}")


if __name__ == "__main__":
    main()
