import math

from ur3e_web_ui.joint_limits import clamp_to_limits, load_ur3e_joint_limits


def test_parses_degrees_tags():
    limits = load_ur3e_joint_limits()
    assert set(limits) == {
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    }
    assert math.isclose(limits["shoulder_pan_joint"].max_position, 2.0 * math.pi)
    assert math.isclose(limits["shoulder_pan_joint"].min_position, -2.0 * math.pi)
    assert math.isclose(limits["shoulder_pan_joint"].max_velocity, math.pi)
    # elbow is artificially limited to +/- pi in ur_description
    assert math.isclose(limits["elbow_joint"].max_position, math.pi)
    assert math.isclose(limits["elbow_joint"].min_position, -math.pi)


def test_wrist3_has_no_position_limits():
    limits = load_ur3e_joint_limits()
    assert not limits["wrist_3_joint"].has_position_limits
    value, clamped = clamp_to_limits("wrist_3_joint", 100.0, limits)
    assert value == 100.0
    assert not clamped


def test_clamping():
    limits = load_ur3e_joint_limits()
    value, clamped = clamp_to_limits("elbow_joint", 4.0, limits)
    assert math.isclose(value, math.pi)
    assert clamped
    value, clamped = clamp_to_limits("elbow_joint", 1.0, limits)
    assert value == 1.0
    assert not clamped
