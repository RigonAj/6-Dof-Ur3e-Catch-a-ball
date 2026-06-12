import json
import math

import pytest

from ur3e_rollout_replay.replay_core import DEFAULT_JOINT_NAMES, ReplayDataError
from ur3e_web_ui.calibration import CalibrationPoseStore, load_camera_calibration

# Shape produced by solve_handeye.py --output-yaml (Dv-Rosws scripts).
SAMPLE_RESULT_YAML = """\
# Hand-eye calibration result (eye-to-hand, UR3e + DVXplorer)
created_at: "2026-06-12T16:03:39.913"
units: "meters"
sample_count: 12
T_base_camera:
  parent: "base"
  child: "camera_optical"
  xyz: [-0.5995011452663233, 1.1206603361887852, -1.4842040863032766]
  quat_xyzw: [0.001186259272833432, 0.28808575933351444, -0.2643561235516248, 0.9203917796237253]
  rpy_rad: [-0.1781033384117694, 0.559698491591448, -0.6106945529292676]
T_tool0_mire:
  parent: "tool0"
  child: "screen_center"
  xyz: [-0.044314877579844425, -0.04902608246917506, -0.010984738823470508]
  quat_xyzw: [0.420563048786509, -0.15445569266665793, -0.19470664613713345, 0.8725591572771949]
validation:
  solver_agreement:
    trans_mm: 2.26e-12
    rot_deg: 0.0
"""

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


def test_load_camera_calibration_parses_solver_yaml(tmp_path):
    path = tmp_path / "handeye_result.yaml"
    path.write_text(SAMPLE_RESULT_YAML)
    data = load_camera_calibration(path)
    transform = data["T_base_camera"]
    assert transform["parent"] == "base"
    assert transform["child"] == "camera_optical"
    assert len(transform["xyz"]) == 3
    assert len(transform["quat_xyzw"]) == 4
    assert data["sample_count"] == 12
    assert data["T_tool0_mire"]["child"] == "screen_center"


def test_load_camera_calibration_rejects_bad_files(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("just: a mapping without transforms\n")
    with pytest.raises(ReplayDataError):
        load_camera_calibration(path)

    path.write_text("T_base_camera:\n  parent: base\n  child: cam\n  xyz: [1, 2]\n  quat_xyzw: [0, 0, 0, 1]\n")
    with pytest.raises(ReplayDataError):
        load_camera_calibration(path)

    with pytest.raises(ReplayDataError):
        load_camera_calibration(tmp_path / "missing.yaml")
