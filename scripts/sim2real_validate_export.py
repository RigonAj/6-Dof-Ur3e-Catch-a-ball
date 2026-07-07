#!/usr/bin/env python3
"""Validate a sim-to-real export directory produced by scripts/skrl/play.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

REQUIRED_FILES = [
    "policy_deterministic.ts",
    "policy_deterministic.onnx",
    "policy_metadata.json",
]

REQUIRED_METADATA_KEYS = [
    "task",
    "checkpoint",
    "dt_s",
    "observation_space",
    "action_space",
    "joint_names",
    "action_semantics",
    "action_delta_scale_rad",
    "joint_velocity_safe_rad_s",
    "joint_acceleration_safe_rad_s2",
    "joint_position_lower_rad",
    "joint_position_upper_rad",
    "rollout_schema_version",
]

REQUIRED_SAMPLE_FIELDS = [
    "step",
    "time_s",
    "observation",
    "action_normalized",
    "joint_position_before_rad",
    "joint_velocity_before_rad_s",
    "joint_position_target_rad",
    "sim_time_after_s",
    "joint_position_after_rad",
    "joint_velocity_after_rad_s",
    "joint_position_target_error_after_rad",
]

LEGACY_ACTION_SEMANTICS = "joint_position_target_rad = action_normalized * action_scale"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TorchScript/ONNX/metadata/rollout artifacts for UR3e sim-to-real V1."
    )
    parser.add_argument("--exports", required=True, help="Export directory produced by scripts/skrl/play.py.")
    parser.add_argument(
        "--allow-legacy-actions",
        action="store_true",
        help="Allow the old absolute action mapping. Default is to reject it for V1 live compatibility.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _require_vector(metadata: dict[str, Any], key: str, length: int, errors: list[str]) -> None:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"metadata.{key} must be a {length}-element list")
        return
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)):
            errors.append(f"metadata.{key}[{index}] must be numeric")


def _validate_metadata(metadata: dict[str, Any], source: str, allow_legacy: bool, errors: list[str]) -> None:
    for key in REQUIRED_METADATA_KEYS:
        _require(key in metadata, f"{source}: missing metadata key {key!r}", errors)

    _require(metadata.get("observation_space") == 33, f"{source}: observation_space must be 33", errors)
    _require(metadata.get("action_space") == 6, f"{source}: action_space must be 6", errors)
    _require(metadata.get("joint_names") == EXPECTED_JOINT_NAMES, f"{source}: unexpected joint_names/order", errors)
    _require(float(metadata.get("dt_s", 0.0) or 0.0) > 0.0, f"{source}: dt_s must be positive", errors)
    _require(int(metadata.get("rollout_schema_version", 0) or 0) >= 2, f"{source}: rollout_schema_version >= 2 required", errors)

    action_semantics = str(metadata.get("action_semantics", ""))
    _require(bool(action_semantics), f"{source}: action_semantics must be non-empty", errors)
    if not allow_legacy:
        _require(
            action_semantics != LEGACY_ACTION_SEMANTICS,
            f"{source}: legacy absolute action semantics are not V1 compatible; regenerate with current play.py",
            errors,
        )

    for key in (
        "action_delta_scale_rad",
        "joint_velocity_safe_rad_s",
        "joint_acceleration_safe_rad_s2",
        "joint_position_lower_rad",
        "joint_position_upper_rad",
    ):
        _require_vector(metadata, key, 6, errors)

    # Optional (older exports lack it): the racket hold side must be a known
    # value and agree with the sign of the disk offset on wrist_3 X.
    hold_side = metadata.get("hold_side")
    if hold_side is not None:
        _require(hold_side in ("right", "left"), f"{source}: hold_side must be 'right' or 'left'", errors)
        disk_offset = metadata.get("disk_offset_wrist_3_link_m")
        if hold_side in ("right", "left") and isinstance(disk_offset, list) and len(disk_offset) == 3:
            expected_sign = -1.0 if hold_side == "right" else 1.0
            _require(
                float(disk_offset[0]) * expected_sign > 0.0,
                f"{source}: disk_offset_wrist_3_link_m x={disk_offset[0]} does not match hold_side={hold_side!r}",
                errors,
            )


def _validate_rollout(path: Path, allow_legacy: bool, errors: list[str]) -> None:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        errors.append(f"{path}: rollout root must be an object")
        return
    metadata = payload.get("metadata")
    episodes = payload.get("episodes")
    if not isinstance(metadata, dict):
        errors.append(f"{path}: missing metadata object")
        return
    _validate_metadata(metadata, f"{path}.metadata", allow_legacy, errors)
    if not isinstance(episodes, list) or not episodes:
        errors.append(f"{path}: episodes must be a non-empty list")
        return

    checked_samples = 0
    for episode_index, episode in enumerate(episodes):
        samples = episode.get("samples") if isinstance(episode, dict) else None
        if not isinstance(samples, list) or not samples:
            errors.append(f"{path}: episode {episode_index} has no samples")
            continue
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                errors.append(f"{path}: episode {episode_index} sample {sample_index} is not an object")
                continue
            missing = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
            if missing:
                errors.append(
                    f"{path}: episode {episode_index} sample {sample_index} missing fields: {', '.join(missing)}"
                )
                continue
            if len(sample["observation"]) != 33:
                errors.append(f"{path}: episode {episode_index} sample {sample_index} observation length != 33")
            for field in (
                "action_normalized",
                "joint_position_before_rad",
                "joint_velocity_before_rad_s",
                "joint_position_target_rad",
                "joint_position_after_rad",
                "joint_velocity_after_rad_s",
                "joint_position_target_error_after_rad",
            ):
                if len(sample[field]) != 6:
                    errors.append(f"{path}: episode {episode_index} sample {sample_index} {field} length != 6")
            checked_samples += 1
    _require(checked_samples > 0, f"{path}: no valid rollout samples checked", errors)


def main() -> None:
    args = _parse_args()
    export_dir = Path(args.exports).expanduser().resolve()
    errors: list[str] = []

    _require(export_dir.is_dir(), f"export directory not found: {export_dir}", errors)
    for filename in REQUIRED_FILES:
        _require((export_dir / filename).is_file(), f"missing required file: {export_dir / filename}", errors)

    metadata_path = export_dir / "policy_metadata.json"
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        if isinstance(metadata, dict):
            _validate_metadata(metadata, str(metadata_path), args.allow_legacy_actions, errors)
        else:
            errors.append(f"{metadata_path}: metadata root must be an object")

    rollout_paths = sorted(export_dir.glob("rollouts_*_episodes.json"))
    _require(bool(rollout_paths), f"no rollouts_*_episodes.json found in {export_dir}", errors)
    for rollout_path in rollout_paths:
        _validate_rollout(rollout_path, args.allow_legacy_actions, errors)

    if errors:
        print("Sim2real export validation FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Sim2real export validation OK: {export_dir}")
    print(f"Checked {len(rollout_paths)} rollout file(s).")


if __name__ == "__main__":
    main()
