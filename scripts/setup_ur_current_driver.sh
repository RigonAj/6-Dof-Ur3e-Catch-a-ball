#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ERROR: /opt/ros/${ROS_DISTRO}/setup.bash not found" >&2
  exit 2
fi

packages=(
  "ros-${ROS_DISTRO}-ur"
  "ros-${ROS_DISTRO}-control-msgs"
  "ros-${ROS_DISTRO}-ros2controlcli"
)

echo "Installing current Universal Robots ROS 2 ${ROS_DISTRO} packages:"
printf '  %s\n' "${packages[@]}"
echo

if [[ -t 0 ]]; then
  sudo -v
elif ! sudo -n true 2>/dev/null; then
  echo "sudo requires a password, and this shell is non-interactive."
  echo "Run this command manually in a terminal, then rerun this script:"
  echo
  echo "  sudo apt-get update && sudo apt-get install -y ${packages[*]}"
  exit 2
fi

sudo apt-get update
sudo apt-get install -y "${packages[@]}"

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

cd "${REPO_ROOT}/ros2_ws"
colcon build --symlink-install

echo
echo "Current UR driver packages installed and replay package built."
echo "Use:"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "  source ${REPO_ROOT}/ros2_ws/install/setup.bash"
