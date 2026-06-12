#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
LEGACY_WS="${UR_LEGACY_WS:-${HOME}/ros2_ur_legacy_ws}"
SRC_DIR="${LEGACY_WS}/src"

echo "WARNING: setup_ur_legacy_driver.sh is deprecated for this project workflow."
echo "Use scripts/setup_ur_current_driver.sh for the current ROS ${ROS_DISTRO} binary driver."
echo

if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ERROR: /opt/ros/${ROS_DISTRO}/setup.bash not found" >&2
  exit 2
fi

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
mkdir -p "${SRC_DIR}"

clone_or_update() {
  local url="$1"
  local ref="$2"
  local dest="$3"

  if [[ ! -d "${dest}/.git" ]]; then
    git clone "${url}" "${dest}"
  fi

  git -C "${dest}" fetch --tags --prune
  git -C "${dest}" checkout "${ref}"
}

clone_or_update \
  "https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git" \
  "2.2.8" \
  "${SRC_DIR}/Universal_Robots_ROS2_Driver"

clone_or_update \
  "https://github.com/UniversalRobots/Universal_Robots_Client_Library.git" \
  "1.3.1" \
  "${SRC_DIR}/Universal_Robots_Client_Library"

clone_or_update \
  "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git" \
  "ros2" \
  "${SRC_DIR}/Universal_Robots_ROS2_Description"

cd "${LEGACY_WS}"
rosdep update
rosdep install --rosdistro "${ROS_DISTRO}" --ignore-src --from-paths src -y
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

echo
echo "Legacy UR driver workspace built."
echo "Source it with:"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "  source ${LEGACY_WS}/install/setup.bash"
