import json
import math

import pytest

from ur3e_rollout_replay.replay_core import DEFAULT_JOINT_NAMES, ReplayDataError
from ur3e_web_ui.calibration import CalibrationPoseStore

POSE_A = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
POSE_B = [0.3, -1.2, 0.5, -1.0, 0.4, 0.1]


def make_store(tmp_path):
    return CalibrationPoseStore(tmp_path / "poses.json")


def test_starts_empty_without_file(tmp_path):
    store = make_store(tmp_path)
    assert store.list_poses() == []
    assert not (tmp_path / "poses.json").exists()


def test_add_persists_and_reloads(tmp_path):
    store = make_store(tmp_path)
    pose = store.add(POSE_A, "tilt_left")
    assert pose["name"] == "tilt_left"
    assert pose["joints_rad"] == POSE_A

    reloaded = make_store(tmp_path)
    assert [p["name"] for p in reloaded.list_poses()] == ["tilt_left"]
    assert reloaded.get(0)["joints_rad"] == POSE_A


def test_auto_names_and_deduplication(tmp_path):
    store = make_store(tmp_path)
    assert store.add(POSE_A)["name"] == "pose_01"
    assert store.add(POSE_B)["name"] == "pose_02"
    assert store.add(POSE_A, "pose_01")["name"] == "pose_01_2"
    assert store.add(POSE_A, "  ")["name"] == "pose_04"


def test_delete_reindexes(tmp_path):
    store = make_store(tmp_path)
    store.add(POSE_A, "first")
    store.add(POSE_B, "second")
    removed = store.delete(0)
    assert removed["name"] == "first"
    assert [p["name"] for p in store.list_poses()] == ["second"]
    with pytest.raises(IndexError):
        store.delete(5)
    with pytest.raises(IndexError):
        store.get(1)


def test_rejects_bad_joint_vectors(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ReplayDataError):
        store.add([0.0, 1.0])
    with pytest.raises(ReplayDataError):
        store.add([math.nan] * len(DEFAULT_JOINT_NAMES))


def test_rejects_corrupt_file(tmp_path):
    path = tmp_path / "poses.json"
    path.write_text("{not json")
    with pytest.raises(ReplayDataError):
        CalibrationPoseStore(path)


def test_rejects_joint_name_mismatch(tmp_path):
    path = tmp_path / "poses.json"
    path.write_text(json.dumps({"joint_names": ["a", "b"], "poses": []}))
    with pytest.raises(ReplayDataError):
        CalibrationPoseStore(path)


def test_file_format_is_documented_schema(tmp_path):
    store = make_store(tmp_path)
    store.add(POSE_A, "p")
    data = json.loads((tmp_path / "poses.json").read_text())
    assert data["joint_names"] == list(DEFAULT_JOINT_NAMES)
    assert data["poses"][0]["name"] == "p"
    assert data["poses"][0]["joints_rad"] == POSE_A
    assert "created_at" in data["poses"][0]
    assert "updated_at" in data
