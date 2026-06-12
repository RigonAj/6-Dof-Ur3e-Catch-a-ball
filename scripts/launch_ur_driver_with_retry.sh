#!/usr/bin/env bash
# Launches the real-robot UR driver, retrying when it hits the known
# ur-client-library 2.11.0 race ("Could not get configuration package
# within timeout"). Proper fix: scripts/upgrade_ur_driver_testing.sh
#
# Usage: ./scripts/launch_ur_driver_with_retry.sh [max_attempts]
# Note: no "set -u" — ROS setup.bash references unbound variables.
set -o pipefail

MAX_ATTEMPTS="${1:-15}"
ROBOT_IP="${UR3E_ROBOT_IP:-192.168.0.5}"
REVERSE_IP="${UR3E_REVERSE_IP:-192.168.0.3}"
CALIB_FILE="${UR3E_CALIBRATION_FILE:-$HOME/ur3e_calibration.yaml}"
LOG_FILE="/tmp/ur_driver_launch.log"

source /opt/ros/humble/setup.bash

EXTRA_ARGS=()
if [[ -f "$CALIB_FILE" ]]; then
  EXTRA_ARGS+=("kinematics_params_file:=$CALIB_FILE")
else
  echo "WARNING: no calibration file at $CALIB_FILE — using default kinematics."
  echo "Extract it with:"
  echo "  ros2 launch ur_calibration calibration_correction.launch.py robot_ip:=$ROBOT_IP target_filename:=$CALIB_FILE"
fi

# SIGTERM, not SIGINT: bash background jobs ignore SIGINT. The launch is
# started in its own process group so retries do not leave helper nodes behind.
stop_launch() {
  local pid="$1"
  if [[ -n "$pid" ]]; then
    kill -TERM "-$pid" 2>/dev/null
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "-$pid" 2>/dev/null
    kill -KILL "$pid" 2>/dev/null
  fi
}

driver_health_ok() {
  timeout 5 ros2 topic echo /joint_states --once >/dev/null 2>&1 &&
    timeout 5 ros2 action info /scaled_joint_trajectory_controller/follow_joint_trajectory |
      grep -qE "Action servers: [1-9]"
}

cleanup() {
  stop_launch "${LAUNCH_PID:-}"
}
trap cleanup EXIT INT TERM

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== Driver launch attempt $attempt/$MAX_ATTEMPTS ==="
  : > "$LOG_FILE"
  setsid ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur3e \
    robot_ip:="$ROBOT_IP" \
    reverse_ip:="$REVERSE_IP" \
    headless_mode:=true \
    launch_rviz:=false \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  LAUNCH_PID=$!

  success=""
  for _ in $(seq 1 40); do  # up to 20 s per attempt
    sleep 0.5
    if grep -q "Could not get configuration package" "$LOG_FILE"; then
      break
    fi
    if grep -q "process has died.*ur_ros2_control_node" "$LOG_FILE"; then
      break
    fi
    if grep -qE "Configured and activated .*scaled_joint_trajectory_controller|configured and activated.*scaled_joint_trajectory_controller" "$LOG_FILE" &&
      driver_health_ok; then
      success=1
      break
    fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      break
    fi
  done

  if [[ -n "$success" ]]; then
    echo "Driver is up (attempt $attempt). Log: $LOG_FILE"
    echo "Press Ctrl+C to stop the driver."
    wait "$LAUNCH_PID"
    exit $?
  fi

  echo "Attempt $attempt failed (client library 2.11 race). Restarting..."
  stop_launch "$LAUNCH_PID"
  wait "$LAUNCH_PID" 2>/dev/null
  LAUNCH_PID=""
  sleep 2
done

echo "ERROR: driver did not come up after $MAX_ATTEMPTS attempts."
echo "Run the permanent fix: sudo ./scripts/upgrade_ur_driver_testing.sh"
exit 1
