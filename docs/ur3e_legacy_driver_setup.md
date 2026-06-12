# UR3e Legacy ROS 2 Driver Setup

Deprecated for the current workflow. Use `docs/ur3e_current_driver_setup.md` unless you explicitly need to keep an old PolyScope version and accept a source-build maintenance path.

This project uses a replay node in `ros2_ws/src/ur3e_rollout_replay` and a separate legacy Universal Robots driver overlay for the physical UR3e.

Your robot details from the first setup pass:

- Robot: UR3e
- Robot IP: `192.168.0.5`
- Robot subnet mask: `255.255.255.0`
- ROS PC wired IP: `192.168.0.3/24`
- PolyScope: `5.5.1.82186`

PolyScope `5.5.1` is older than the current UR client library compatibility line. For physical robot control, use a legacy source build pinned to:

- `Universal_Robots_ROS2_Driver`: `2.2.8`
- `Universal_Robots_Client_Library`: `1.3.1`
- `Universal_Robots_ROS2_Description`: `ros2`

## 1. Fix The Wired Network

The current wired interface was observed as `192.168.0.3/0`, which routes `192.168.0.5` through Wi-Fi instead of Ethernet.

Preview the NetworkManager commands:

```bash
./scripts/configure_ur3e_ethernet.sh
```

Apply them:

```bash
./scripts/configure_ur3e_ethernet.sh --apply
```

Then verify:

```bash
ip -br addr show enx00e04c3211b0
ip route get 192.168.0.5
ping -c 3 192.168.0.5
```

`ip route get 192.168.0.5` should use `enx00e04c3211b0`, not Wi-Fi.

## 2. Build The Legacy Driver Overlay

Run:

```bash
./scripts/setup_ur_legacy_driver.sh
```

The script creates `~/ros2_ur_legacy_ws`, clones the pinned sources, installs ROS dependencies with `rosdep`, and builds with `colcon`.

Source it when you need the UR driver:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ur_legacy_ws/install/setup.bash
```

## 3. Build The Replay Package

From this repository:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ur_legacy_ws/install/setup.bash
cd ros2_ws
rosdep install --rosdistro humble --ignore-src --from-paths src -y
colcon build --symlink-install
source install/setup.bash
```

## 4. Mock-Hardware Acceptance

Terminal 1:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ur_legacy_ws/install/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.0.5 \
  use_fake_hardware:=true \
  launch_rviz:=false
```

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ur_legacy_ws/install/setup.bash
source ros2_ws/install/setup.bash
ros2 control list_controllers
ros2 run ur3e_rollout_replay ur3e_replay_validate
ros2 run ur3e_rollout_replay ur3e_replay_send --current-joints 0,0,0,0,0,0
ros2 run ur3e_rollout_replay ur3e_replay_send --current-joints 0,0,0,0,0,0 --execute
```

The first `ur3e_replay_send` is a dry run. It prints the retimed trajectory summary and does not send a goal.

## 5. Physical Robot Gate

Only move the real UR3e after mock hardware succeeds.

- Install External Control URCap `v1.0.5`.
- In the URCap installation, set the remote PC IP to `192.168.0.3`.
- Extract calibration:

```bash
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=192.168.0.5 \
  target_filename:=$HOME/ur3e_calibration.yaml
```

- Launch the real driver:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.0.5 \
  reverse_ip:=192.168.0.3 \
  kinematics_params_file:=$HOME/ur3e_calibration.yaml \
  launch_rviz:=true
```

- Press play on the External Control program on the teach pendant.
- Remove the ball and obstacles, use reduced speed, and keep an operator at the E-stop.
- Run `ros2 run ur3e_rollout_replay ur3e_replay_send` without `--execute` first, then use `--execute` only after reviewing the printed timing.
