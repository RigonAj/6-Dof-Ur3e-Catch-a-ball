from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Sequence

import yaml

from ur3e_rollout_replay.replay_core import DEFAULT_JOINT_NAMES, ReplayDataError

# Relative to the process working directory (the launcher runs from the repo
# root), so recorded poses are versioned alongside the calibration data.
DEFAULT_POSES_PATH = Path("calibration/calibration_poses.json")
# Result written by solve_handeye.py --output-yaml (Dv-Rosws scripts).
DEFAULT_CAMERA_RESULT_PATH = Path("calibration/handeye_result.yaml")

_TRANSFORM_KEYS = ("parent", "child", "xyz", "quat_xyzw")


def load_camera_calibration(path: Path | str) -> dict:
    """Parse the hand-eye result YAML; raises ReplayDataError when invalid.

    Returns the raw document; T_base_camera is validated (the viewer needs
    it), T_tool0_mire and validation are passed through when present.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReplayDataError(f"cannot read camera calibration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReplayDataError(f"{path}: not a YAML mapping")
    transform = data.get("T_base_camera")
    if not isinstance(transform, dict):
        raise ReplayDataError(f"{path}: missing T_base_camera")
    for key in _TRANSFORM_KEYS:
        if key not in transform:
            raise ReplayDataError(f"{path}: T_base_camera is missing '{key}'")
    if len(transform["xyz"]) != 3 or len(transform["quat_xyzw"]) != 4:
        raise ReplayDataError(f"{path}: T_base_camera needs xyz[3] and quat_xyzw[4]")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in (*transform["xyz"], *transform["quat_xyzw"])
    ):
        raise ReplayDataError(f"{path}: T_base_camera values must be finite numbers")
    return data


class CalibrationPoseStore:
    """Named joint configurations for the hand-eye calibration session.

    Poses are stored in joint space on purpose: IK has multiple branches and
    home sits at a wrist singularity, so Cartesian targets are not repeatable
    across sessions (docs/ur3e_camera_base_calibration.md section 7). Saved
    once, replayed identically.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_POSES_PATH,
        joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
    ) -> None:
        self.path = Path(path)
        self.joint_names = tuple(joint_names)
        self._lock = Lock()
        self._poses: list[dict] = []
        self._load()

    # ------------------------------------------------------------------ io

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayDataError(f"cannot read calibration poses {self.path}: {exc}") from exc
        names = data.get("joint_names")
        if names != list(self.joint_names):
            raise ReplayDataError(
                f"{self.path}: joint_names mismatch (file {names}, expected {list(self.joint_names)})"
            )
        poses = data.get("poses")
        if not isinstance(poses, list):
            raise ReplayDataError(f"{self.path}: poses must be a list")
        for index, pose in enumerate(poses):
            self._validate_pose(pose, index)
        self._poses = poses

    def _validate_pose(self, pose: object, index: int) -> None:
        if not isinstance(pose, dict) or not isinstance(pose.get("name"), str):
            raise ReplayDataError(f"{self.path}: pose {index} must have a string name")
        joints = pose.get("joints_rad")
        if (
            not isinstance(joints, list)
            or len(joints) != len(self.joint_names)
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in joints)
        ):
            raise ReplayDataError(
                f"{self.path}: pose {index} needs {len(self.joint_names)} finite joints_rad"
            )

    def _save_locked(self) -> None:
        payload = {
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
            "joint_names": list(self.joint_names),
            "poses": self._poses,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    # ------------------------------------------------------------------ api

    def list_poses(self) -> list[dict]:
        with self._lock:
            return [dict(pose) for pose in self._poses]

    def get(self, index: int) -> dict:
        with self._lock:
            if not 0 <= index < len(self._poses):
                raise IndexError(f"pose index {index} out of range (have {len(self._poses)})")
            return dict(self._poses[index])

    def add(self, joints_rad: Sequence[float], name: str | None = None) -> dict:
        joints = [float(value) for value in joints_rad]
        if len(joints) != len(self.joint_names) or not all(math.isfinite(v) for v in joints):
            raise ReplayDataError(f"a pose needs {len(self.joint_names)} finite joint values")
        with self._lock:
            clean = (name or "").strip()
            if not clean:
                clean = f"pose_{len(self._poses) + 1:02d}"
            if any(pose["name"] == clean for pose in self._poses):
                suffix = 2
                while any(pose["name"] == f"{clean}_{suffix}" for pose in self._poses):
                    suffix += 1
                clean = f"{clean}_{suffix}"
            pose = {
                "name": clean,
                "joints_rad": joints,
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            self._poses.append(pose)
            self._save_locked()
            return dict(pose)

    def delete(self, index: int) -> dict:
        with self._lock:
            if not 0 <= index < len(self._poses):
                raise IndexError(f"pose index {index} out of range (have {len(self._poses)})")
            removed = self._poses.pop(index)
            self._save_locked()
            return dict(removed)
