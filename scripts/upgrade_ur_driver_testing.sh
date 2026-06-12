#!/usr/bin/env bash
# Fix for: "Could not get configuration package within timeout (1 sec)"
#
# Humble stable ships ur-client-library 2.11.0, which has a too-short primary
# protocol timeout and fails to connect to the robot ~9 out of 10 launches.
# UR fixed this in client library 2.12.0, available from the ros2-testing apt
# repository. Keep the rest of the UR driver stack on the stable repo versions:
# ros2-testing also contains newer ur_description/controller builds that expect
# driver interfaces not exported by the stable Humble ur_robot_driver package.
# See:
# https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/issues/1802
#
# Run with: sudo ./scripts/upgrade_ur_driver_testing.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo $0"
  exit 1
fi

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2-testing/ubuntu jammy main" \
  > /etc/apt/sources.list.d/ros2-testing.list

apt-get update

# Upgrade only the communication library, then force the packages that describe
# controller/hardware interfaces back to the stable versions that match the
# stable ur_robot_driver binary.
apt-get install -y --allow-downgrades \
  ros-humble-ur-client-library=2.12.0-1jammy.20260519.184707 \
  ros-humble-ur-controllers=2.13.0-1jammy.20260505.184649 \
  ros-humble-ur-dashboard-msgs=2.13.0-1jammy.20260414.032926 \
  ros-humble-ur-description=2.10.0-1jammy.20260422.111151 \
  ros-humble-ur-msgs=2.5.0-1jammy.20260414.140739 \
  ros-humble-ur-robot-driver=2.13.0-1jammy.20260505.190135 \
  ros-humble-ur-calibration=2.13.0-1jammy.20260505.190829 \
  ros-humble-ur=2.13.0-1jammy.20260505.200709

echo
echo "Installed versions now:"
dpkg-query -W \
  ros-humble-ur-client-library \
  ros-humble-ur-controllers \
  ros-humble-ur-description \
  ros-humble-ur-robot-driver
echo
echo "Done. Relaunch the driver normally."
