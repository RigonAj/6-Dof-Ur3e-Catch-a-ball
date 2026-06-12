# UR3e Current ROS 2 Humble Driver Setup

Use this path when you want the current binary Universal Robots driver for ROS 2 Humble instead of the legacy source build.

Known local setup:
--------
- Robot: UR3e
- Robot IP: `192.168.0.5`
- Robot subnet mask: `255.255.255.0`
- ROS PC wired IP: `192.168.0.3/24`
- Wired interface observed locally: `enx00e04c3211b0`
- Installed driver: `ros-humble-ur` `2.13.0`
- PolyScope: `5.12.4` (updated 2026-06; previously `5.5.1.82186`)

PolyScope 5.12.4 is compatible with the binary `ros-humble-ur` 2.13.0 driver — no clean reinstall is needed. The legacy source-build path in `docs/ur3e_legacy_driver_setup.md` is obsolete and kept only for history.

Robot-side one-time setup on PolyScope 5.12.4:

1. Install the **External Control URCap** (Settings → System → URCaps, load from USB).
2. In the URCap installation settings, set the remote host IP to `192.168.0.3` (port `50002`).
3. Create a program containing a single ExternalControl node and save it.
4. Enable Remote Control: Settings → System → Remote Control → Enable.

For a browser-based control panel (live 3D model, jog, TCP pose, rollout replay), see `docs/ur3e_web_ui.md`.

For the full history of the "commands accepted but robot does not move" issue and its resolution, see `docs/ur3e_motion_issue_resolution.md`.

## 1. Install Current Humble Driver Packages

Run this from the repo root:

```bash
./scripts/setup_ur_current_driver.sh
```

The script installs:

```text
ros-humble-ur
ros-humble-control-msgs
ros-humble-ros2controlcli
```

It also rebuilds this repo's replay package in `ros2_ws`.

If sudo is not available in the current terminal, run the install part manually:

```bash
sudo apt-get update
sudo apt-get install -y ros-humble-ur ros-humble-control-msgs ros-humble-ros2controlcli
```

Then build the replay package:

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 2. Fix The Wired Network

Preview:

```bash
./scripts/configure_ur3e_ethernet.sh
```

Apply:

```bash
./scripts/configure_ur3e_ethernet.sh --apply
```

Verify:

```bash
ip route get 192.168.0.5
ping -c 3 192.168.0.5
```

The route should use `enx00e04c3211b0` with source `192.168.0.3`.

## 3. Mock-Hardware Test

Terminal 1:

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.0.5 \
  use_fake_hardware:=true \
  launch_rviz:=false
```

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 control list_controllers
ros2 run ur3e_rollout_replay ur3e_replay_validate
ros2 run ur3e_rollout_replay ur3e_replay_send --current-joints 0,0,0,0,0,0
ros2 run ur3e_rollout_replay ur3e_replay_send --current-joints 0,0,0,0,0,0 --execute
```

The first `ur3e_replay_send` is a dry run. It prints the retimed trajectory summary and does not send a goal.

## 4. Physical Robot Gate

Only move the real UR3e after the mock-hardware test succeeds.

- Confirm the current driver supports the robot's PolyScope version, or update PolyScope first.
- Install/configure the External Control URCap required by the current driver.
- Set the URCap remote PC IP to `192.168.0.3`.
- Remove the ball and unnecessary obstacles.
- Use reduced mode or a low teach-pendant speed slider.
- Keep an operator at the E-stop.

Extract calibration:

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=192.168.0.5 \
  target_filename:=$HOME/ur3e_calibration.yaml
```

Launch the real driver:

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.0.5 \
  reverse_ip:=192.168.0.3 \
  kinematics_params_file:=$HOME/ur3e_calibration.yaml \
  launch_rviz:=true
```

Press play on the External Control program on the teach pendant. In another terminal:

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 control list_controllers
ros2 run ur3e_rollout_replay ur3e_replay_send
```

Review the dry-run trajectory output first. Add `--execute` only after the real robot is safe to move.

## Known Issue: "Could not get configuration package within timeout"

With the Humble stable packages (`ur-client-library` 2.11.0), the driver's hardware
interface intermittently fails to start against the real robot with:

```text
[FATAL] [URPositionHardwareInterface]: Could not get configuration package within
timeout, are you connected to the robot?(Configured timeout: 1 sec)
```

The dashboard and RTDE connect fine; only the primary-interface configuration read
times out. This is a known upstream bug fixed in `ur-client-library` 2.12.0
(https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/issues/1802).

Permanent fix (adds the ros2-testing apt repo and upgrades the UR stack):

```bash
sudo ./scripts/upgrade_ur_driver_testing.sh
```

Interim workaround (the failure is a race, so retry until it connects):

```bash
./scripts/launch_ur_driver_with_retry.sh
```

The retry script launches in headless mode with the extracted calibration and
restarts the driver automatically until the hardware interface configures.
