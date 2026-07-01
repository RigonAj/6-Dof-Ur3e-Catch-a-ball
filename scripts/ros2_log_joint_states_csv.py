#!/usr/bin/env python3
"""Log UR joint states from ROS 2 to a simple CSV for sim/real comparison."""

from __future__ import annotations

import argparse
import csv

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log /joint_states to CSV for UR3e sim/real rollout comparison.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--topic", default="/joint_states", help="JointState topic to subscribe to.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional recording duration in seconds. 0 means until Ctrl-C.",
    )
    return parser.parse_args()


class JointStateCsvLogger(Node):
    def __init__(self, output_path: str, topic: str, duration: float):
        super().__init__("ur3e_joint_state_csv_logger")
        self.duration = duration
        self.start_time = None
        self.file = open(output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time_s", *JOINT_NAMES])
        self.subscription = self.create_subscription(JointState, topic, self._on_joint_state, 10)
        self.get_logger().info(f"Logging {topic} to {output_path}")

    def _message_time(self, msg: JointState) -> float:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
        if stamp == 0.0:
            stamp = self.get_clock().now().nanoseconds * 1.0e-9
        if self.start_time is None:
            self.start_time = stamp
        return stamp - self.start_time

    def _on_joint_state(self, msg: JointState) -> None:
        positions_by_name = dict(zip(msg.name, msg.position, strict=False))
        if any(name not in positions_by_name for name in JOINT_NAMES):
            return
        time_s = self._message_time(msg)
        self.writer.writerow([f"{time_s:.9f}", *[f"{positions_by_name[name]:.12f}" for name in JOINT_NAMES]])
        self.file.flush()
        if self.duration > 0.0 and time_s >= self.duration:
            self.get_logger().info("Requested duration reached")
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        self.file.close()
        return super().destroy_node()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = JointStateCsvLogger(args.output, args.topic, args.duration)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
