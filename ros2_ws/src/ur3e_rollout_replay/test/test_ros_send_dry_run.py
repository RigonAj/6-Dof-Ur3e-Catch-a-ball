from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import pytest


def test_send_node_dry_run_with_current_joint_override() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("control_msgs.action")

    from ur3e_rollout_replay.send import main

    result = main(
        [
            "--episode",
            "0",
            "--current-joints",
            "0,0,0,0,0,0",
        ]
    )

    assert result == 0


def test_send_node_execute_with_fake_action_server() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("control_msgs.action")

    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionServer
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    received: queue.Queue = queue.Queue()

    class FakeTrajectoryServer(Node):
        def __init__(self) -> None:
            super().__init__("fake_scaled_joint_trajectory_controller")
            self.server = ActionServer(
                self,
                FollowJointTrajectory,
                "/scaled_joint_trajectory_controller/follow_joint_trajectory",
                self.execute_callback,
            )

        def execute_callback(self, goal_handle):
            received.put(goal_handle.request.trajectory)
            goal_handle.succeed()
            result = FollowJointTrajectory.Result()
            result.error_code = 0
            return result

    rclpy.init(args=None)
    server_node = FakeTrajectoryServer()
    executor = MultiThreadedExecutor()
    executor.add_node(server_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        env = os.environ.copy()
        source_root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = f"{source_root}:{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ur3e_rollout_replay.send",
                "--episode",
                "0",
                "--current-joints",
                "0,0,0,0,0,0",
                "--execute",
            ],
            check=False,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15.0,
        )

        assert result.returncode == 0, result.stdout
        trajectory = received.get(timeout=2.0)
        assert trajectory.joint_names == [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        assert len(trajectory.points) == 12
        # Real trajectory_msgs fields are array('d'), not list
        assert list(trajectory.points[0].positions) == [0.0] * 6
        assert trajectory.points[1].time_from_start.sec >= 10
        assert list(trajectory.points[0].velocities) == []
        assert list(trajectory.points[0].accelerations) == []
    finally:
        executor.shutdown()
        server_node.destroy_node()
        rclpy.shutdown()
