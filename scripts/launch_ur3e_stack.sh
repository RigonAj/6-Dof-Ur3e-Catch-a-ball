#!/usr/bin/env bash
# Launch the real UR3e ROS driver and the web UI in the right order.
#
# Defaults match docs/ur3e_current_driver_setup.md:
#   robot: 192.168.0.5, ROS PC/reverse IP: 192.168.0.3
#
# Ctrl+C stops the UI first, then the driver, including child processes.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_IP="${UR3E_ROBOT_IP:-192.168.0.5}"
REVERSE_IP="${UR3E_REVERSE_IP:-192.168.0.3}"
CALIB_FILE="${UR3E_CALIBRATION_FILE:-$HOME/ur3e_calibration.yaml}"
UI_HOST="${UR3E_UI_HOST:-127.0.0.1}"
UI_PORT="${UR3E_UI_PORT:-8080}"
DRIVER_LOG="${UR3E_DRIVER_LOG:-/tmp/ur3e_driver.log}"
UI_LOG="${UR3E_UI_LOG:-/tmp/ur3e_web_ui.log}"
MOVEIT_LOG="${UR3E_MOVEIT_LOG:-/tmp/ur3e_moveit.log}"
DRIVER_READY_TIMEOUT="${UR3E_DRIVER_READY_TIMEOUT:-45}"
MOVEIT_READY_TIMEOUT="${UR3E_MOVEIT_READY_TIMEOUT:-45}"
UI_READY_TIMEOUT="${UR3E_UI_READY_TIMEOUT:-15}"
HEADLESS_MODE="${UR3E_HEADLESS_MODE:-false}"
CLEAN_EXISTING="${UR3E_CLEAN_EXISTING:-true}"
LAUNCH_MOVEIT="${UR3E_LAUNCH_MOVEIT:-true}"
MAX_JOINT_VELOCITY="${UR3E_MAX_JOINT_VELOCITY:-0.25}"
MAX_JOINT_ACCELERATION="${UR3E_MAX_JOINT_ACCELERATION:-0.5}"
APPROACH_MIN_DURATION="${UR3E_APPROACH_MIN_DURATION:-10.0}"
MIN_SEGMENT_DURATION="${UR3E_MIN_SEGMENT_DURATION:-0.5}"

DRIVER_PID=""
MOVEIT_PID=""
UI_PID=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --stop              Stop existing UR3e driver/UI processes and exit.
  --no-clean          Do not stop existing UR3e driver/UI processes before launch.
  --headless          Launch the UR driver with headless_mode:=true.
  --no-moveit        Do not launch MoveIt /compute_ik.
  --host HOST         UI host, default: $UI_HOST
  --port PORT         UI port, default: $UI_PORT
  -h, --help          Show this help.

Environment:
  UR3E_ROBOT_IP, UR3E_REVERSE_IP, UR3E_CALIBRATION_FILE
  UR3E_UI_HOST, UR3E_UI_PORT, UR3E_HEADLESS_MODE, UR3E_LAUNCH_MOVEIT
  UR3E_MAX_JOINT_VELOCITY, UR3E_MAX_JOINT_ACCELERATION
  UR3E_APPROACH_MIN_DURATION, UR3E_MIN_SEGMENT_DURATION
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop)
      STOP_ONLY=true
      shift
      ;;
    --no-clean)
      CLEAN_EXISTING=false
      shift
      ;;
    --headless)
      HEADLESS_MODE=true
      shift
      ;;
    --no-moveit)
      LAUNCH_MOVEIT=false
      shift
      ;;
    --host)
      UI_HOST="$2"
      shift 2
      ;;
    --port)
      UI_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

source /opt/ros/humble/setup.bash
if [[ -f "$REPO_ROOT/ros2_ws/install/setup.bash" ]]; then
  source "$REPO_ROOT/ros2_ws/install/setup.bash"
else
  echo "ERROR: missing $REPO_ROOT/ros2_ws/install/setup.bash" >&2
  echo "Build first: cd $REPO_ROOT/ros2_ws && colcon build --symlink-install" >&2
  exit 1
fi

stop_group() {
  local pid="$1"
  local label="$2"
  local pgid

  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0

  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  echo "Stopping $label (pid=$pid, pgid=${pgid:-unknown})..."
  if [[ -n "$pgid" && "$pgid" != "$(ps -o pgid= -p $$ | tr -d ' ')" ]]; then
    kill -TERM "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi

  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.5
  done

  if [[ -n "$pgid" && "$pgid" != "$(ps -o pgid= -p $$ | tr -d ' ')" ]]; then
    kill -KILL "-$pgid" 2>/dev/null || true
  else
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

matching_pids() {
  ps -eo pid=,cmd= |
    awk '
      /ros2 launch ur_robot_driver ur_control.launch.py/ ||
      /\/opt\/ros\/humble\/lib\/ur_robot_driver\/ur_ros2_control_node/ ||
      /\/opt\/ros\/humble\/lib\/ur_robot_driver\/dashboard_client/ ||
      /\/opt\/ros\/humble\/lib\/ur_robot_driver\/robot_state_helper/ ||
      /\/opt\/ros\/humble\/lib\/ur_robot_driver\/controller_stopper_node/ ||
      /\/opt\/ros\/humble\/lib\/ur_robot_driver\/urscript_interface/ ||
      /\/opt\/ros\/humble\/lib\/robot_state_publisher\/robot_state_publisher/ ||
      /\/opt\/ros\/humble\/lib\/ur_robot_driver\/trajectory_until_node/ ||
      /ros2 launch ur_moveit_config ur_moveit.launch.py/ ||
      /\/opt\/ros\/humble\/lib\/moveit_ros_move_group\/move_group/ ||
      /ros2 run ur3e_web_ui ur3e_web_ui/ ||
      /\/ur3e_web_ui\/lib\/ur3e_web_ui\/ur3e_web_ui/ {
        if ($0 !~ /awk / && $0 !~ /launch_ur3e_stack/) print $1
      }'
}

stop_existing_stack() {
  local pids
  pids="$(matching_pids || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "Stopping existing UR3e driver/UI processes..."
  # Kill individual matched pids instead of their whole process groups: old
  # interactive launches may share a terminal process group with unrelated jobs.
  kill -TERM $pids 2>/dev/null || true
  sleep 2

  pids="$(matching_pids || true)"
  if [[ -n "$pids" ]]; then
    kill -KILL $pids 2>/dev/null || true
  fi

  ros2 daemon stop >/dev/null 2>&1 || true
  ros2 daemon start >/dev/null 2>&1 || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_group "$UI_PID" "UR3e web UI"
  stop_group "$MOVEIT_PID" "MoveIt"
  stop_group "$DRIVER_PID" "UR3e driver"
  exit "$status"
}
trap cleanup EXIT INT TERM

driver_health_ok() {
  timeout 4 ros2 topic echo /joint_states --once >/dev/null 2>&1 &&
    timeout 4 ros2 action info /scaled_joint_trajectory_controller/follow_joint_trajectory |
      grep -qE "Action servers: [1-9]"
}

wait_for_driver() {
  local deadline=$((SECONDS + DRIVER_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if driver_health_ok; then
      return 0
    fi
    if [[ -n "$DRIVER_PID" ]] && ! kill -0 "$DRIVER_PID" 2>/dev/null; then
      return 1
    fi
    if grep -qE "process has died.*ur_ros2_control_node|Could not get configuration package" "$DRIVER_LOG" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

moveit_health_ok() {
  local services
  services="$(timeout 4 ros2 service list 2>/dev/null || true)"
  grep -qx "/compute_ik" <<< "$services"
}

wait_for_moveit() {
  local deadline=$((SECONDS + MOVEIT_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if moveit_health_ok; then
      return 0
    fi
    if [[ -n "$MOVEIT_PID" ]] && ! kill -0 "$MOVEIT_PID" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

wait_for_ui() {
  local deadline=$((SECONDS + UI_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://$UI_HOST:$UI_PORT/api/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$UI_PID" ]] && ! kill -0 "$UI_PID" 2>/dev/null; then
      return 1
    fi
    sleep 0.5
  done
  return 1
}

if [[ "${STOP_ONLY:-false}" == "true" ]]; then
  stop_existing_stack
  trap - EXIT INT TERM
  echo "UR3e driver/UI stopped."
  exit 0
fi

if [[ "$CLEAN_EXISTING" == "true" ]]; then
  stop_existing_stack
fi

if ! ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; then
  echo "ERROR: robot $ROBOT_IP is not reachable. Check Ethernet before launching." >&2
  exit 1
fi

if ss -ltn "sport = :$UI_PORT" | grep -q ":$UI_PORT"; then
  echo "ERROR: port $UI_PORT is already in use. Stop the old UI or pass --port." >&2
  exit 1
fi

driver_args=(
  ros2 launch ur_robot_driver ur_control.launch.py
  "ur_type:=ur3e"
  "robot_ip:=$ROBOT_IP"
  "reverse_ip:=$REVERSE_IP"
  "launch_rviz:=false"
)
if [[ "$HEADLESS_MODE" == "true" ]]; then
  driver_args+=("headless_mode:=true")
fi
if [[ -f "$CALIB_FILE" ]]; then
  driver_args+=("kinematics_params_file:=$CALIB_FILE")
else
  echo "WARNING: no calibration file at $CALIB_FILE; using default kinematics."
fi

echo "Launching UR3e driver..."
echo "  robot_ip=$ROBOT_IP reverse_ip=$REVERSE_IP headless=$HEADLESS_MODE"
echo "  log: $DRIVER_LOG"
: > "$DRIVER_LOG"
setsid "${driver_args[@]}" > "$DRIVER_LOG" 2>&1 &
DRIVER_PID=$!

if ! wait_for_driver; then
  echo "ERROR: UR3e driver did not become healthy." >&2
  echo "Last driver log lines:" >&2
  tail -80 "$DRIVER_LOG" >&2 || true
  exit 1
fi
echo "UR3e driver is healthy."

if [[ "$LAUNCH_MOVEIT" == "true" ]]; then
  echo "Launching MoveIt IK service..."
  echo "  log: $MOVEIT_LOG"
  : > "$MOVEIT_LOG"
  setsid ros2 launch ur_moveit_config ur_moveit.launch.py \
    "ur_type:=ur3e" \
    "launch_rviz:=false" \
    "launch_servo:=false" \
    > "$MOVEIT_LOG" 2>&1 &
  MOVEIT_PID=$!

  if wait_for_moveit; then
    echo "MoveIt IK service is healthy."
  else
    echo "WARNING: MoveIt /compute_ik did not become healthy; TCP target IK will be unavailable." >&2
    echo "Last MoveIt log lines:" >&2
    tail -80 "$MOVEIT_LOG" >&2 || true
  fi
fi

echo "Launching UR3e web UI..."
echo "  url: http://$UI_HOST:$UI_PORT"
echo "  replay limits: vel=$MAX_JOINT_VELOCITY rad/s acc=$MAX_JOINT_ACCELERATION rad/s^2"
echo "  replay timing: approach=$APPROACH_MIN_DURATION s min_segment=$MIN_SEGMENT_DURATION s"
echo "  log: $UI_LOG"
: > "$UI_LOG"
setsid ros2 run ur3e_web_ui ur3e_web_ui \
  --host "$UI_HOST" \
  --port "$UI_PORT" \
  --max-joint-velocity "$MAX_JOINT_VELOCITY" \
  --max-joint-acceleration "$MAX_JOINT_ACCELERATION" \
  --approach-min-duration "$APPROACH_MIN_DURATION" \
  --min-segment-duration "$MIN_SEGMENT_DURATION" \
  > "$UI_LOG" 2>&1 &
UI_PID=$!

if ! wait_for_ui; then
  echo "ERROR: UR3e web UI did not become healthy." >&2
  echo "Last UI log lines:" >&2
  tail -80 "$UI_LOG" >&2 || true
  exit 1
fi

echo
echo "UR3e stack is running:"
echo "  UI: http://$UI_HOST:$UI_PORT"
echo "  Driver log: $DRIVER_LOG"
if [[ -n "$MOVEIT_PID" ]]; then
  echo "  MoveIt log: $MOVEIT_LOG"
fi
echo "  UI log: $UI_LOG"
echo
echo "Press Ctrl+C here to stop the UI and driver cleanly."

misses=0
while true; do
  sleep 5

  if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    echo "ERROR: UR3e driver launch process exited." >&2
    exit 1
  fi
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    echo "ERROR: UR3e web UI exited." >&2
    tail -80 "$UI_LOG" >&2 || true
    exit 1
  fi
  if [[ -n "$MOVEIT_PID" ]] && ! kill -0 "$MOVEIT_PID" 2>/dev/null; then
    echo "WARNING: MoveIt process exited; TCP target IK is unavailable."
    MOVEIT_PID=""
  fi

  if driver_health_ok; then
    misses=0
  else
    misses=$((misses + 1))
    echo "WARNING: UR3e driver health check missed ($misses/3)."
    if (( misses >= 3 )); then
      echo "ERROR: UR3e driver is no longer healthy." >&2
      tail -80 "$DRIVER_LOG" >&2 || true
      exit 1
    fi
  fi
done
