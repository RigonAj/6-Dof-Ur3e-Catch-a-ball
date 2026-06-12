# UR3e Motion Issue Resolution

Date: 2026-06-11

This note documents the initial real-robot problem, the diagnosis, and the fixes applied to make the UR3e accept ROS 2 driver motion commands and then expose safer replay controls in the web UI.

## Initial Problem

The robot was reachable and partially controllable, but motion commands did not move the arm:

- The browser UI showed live joint positions from `/joint_states`.
- Dashboard commands such as Power On / Power Off / Brake Release worked.
- ROS action goals could be sent to `/scaled_joint_trajectory_controller/follow_joint_trajectory`.
- The robot did not physically execute jog/home/replay movements.

This was confusing because networking and the Dashboard server were clearly alive. The important detail is that Dashboard, RTDE joint-state streaming, and trajectory execution are separate paths. Dashboard commands can work while the External Control program is stopped or while speed scaling is `0%`.

## Robot And Network Setup

Known working setup:

```text
Robot: UR3e
Robot IP: 192.168.0.5
Robot subnet mask: 255.255.255.0
ROS PC wired IP / reverse IP: 192.168.0.3
Calibration file: $HOME/ur3e_calibration.yaml
PolyScope: 5.12.4
ROS 2: Humble on Ubuntu 22.04
```

The robot side must have:

1. External Control URCap installed.
2. External Control remote host set to `192.168.0.3`.
3. A program containing an ExternalControl node.
4. Remote Control enabled.
5. The External Control program actively playing before sending motion goals.
6. Teach-pendant speed slider above `0%`.

## Root Causes Found

### 1. UR Driver Package Version Problem

The original driver stack had a known failure mode around the UR client library:

```text
Could not get configuration package within timeout, are you connected to the robot?
```

The fix was to use the current binary ROS 2 Humble UR driver stack, with `ur-client-library` upgraded to a version containing the connection fix, while keeping the rest of the UR packages compatible.

Final working package set observed locally:

```bash
dpkg-query -W \
  ros-humble-ur-client-library \
  ros-humble-ur-description \
  ros-humble-ur-controllers \
  ros-humble-ur-robot-driver \
  ros-humble-ur
```

Expected output:

```text
ros-humble-ur                    2.13.0-1jammy.20260505.200709
ros-humble-ur-client-library     2.12.0-1jammy.20260519.184707
ros-humble-ur-controllers        2.13.0-1jammy.20260505.184649
ros-humble-ur-description        2.10.0-1jammy.20260422.111151
ros-humble-ur-robot-driver       2.13.0-1jammy.20260505.190135
```

Important: mixing too many UR packages from `ros2-testing` can create interface/description mismatches. The stable driver packages above, plus `ur-client-library` `2.12.0`, were the working combination.

### 2. External Control / Speed Scaling Was The Motion Gate

After the driver package issue was fixed, the UI could still show this state:

- live joint positions OK
- action server ready OK
- Dashboard commands OK
- motion commands accepted but no physical movement

The key diagnostic topic was:

```bash
ros2 topic echo /speed_scaling_state_broadcaster/speed_scaling --once
```

When this was `0.0`, motion goals would not progress. The physical fix is:

1. Press Play on the External Control program on the teach pendant, or use Dashboard Play when available.
2. Raise the teach-pendant speed slider above `0%`.
3. Confirm that the UI `program` badge is running and the `speed` badge is greater than `0%`.

The web UI was updated so jog/home/replay commands are rejected early with a clear message when:

- External Control is stopped.
- speed scaling is `0%`.

This avoids the misleading state where the ROS action server accepts a goal but the robot does not move.

## Fixes Applied In This Repo

### One-Command Launcher

Added:

```text
scripts/launch_ur3e_stack.sh
```

And aliases in `script.zsh`:

```bash
alias ur3e_stack="./scripts/launch_ur3e_stack.sh"
alias ur3e_stop="./scripts/launch_ur3e_stack.sh --stop"
```

The launcher now:

1. Stops stale UR driver/UI processes.
2. Sources ROS and the workspace install.
3. Checks robot reachability.
4. Starts `ur_robot_driver` with the correct `robot_ip`, `reverse_ip`, and calibration file.
5. Waits for `/joint_states`.
6. Waits for `/scaled_joint_trajectory_controller/follow_joint_trajectory`.
7. Starts the web UI on `http://127.0.0.1:8080`.
8. Cleans up UI and driver processes on `Ctrl+C`.

Current launch command:

```bash
source script.zsh
ur3e_stack
```

Current stop command:

```bash
ur3e_stop
```

### Web UI Motion Gating

Updated the web UI backend and frontend to check robot execution readiness:

- `/api/jog`
- `/api/move_home`
- `/api/rollout/{i}/execute`

These now refuse motion if the robot reports:

```text
External Control stopped
```

or:

```text
speed scaling is 0%
```

The UI also shows warning badges/banner for these states.

### Replay Timing Controls

Replay felt slow because the recorded Isaac trajectories were retimed using conservative safety limits and an approach segment was prepended from the robot's current physical pose to the first recorded replay pose.

The UI now exposes replay timing controls in the Rollout tab:

- Max velocity
- Max acceleration
- Approach min
- Segment min
- Safe / Balanced / Fast presets
- Include live approach when validating/previews

The teach-pendant speed slider and UR controller limits still apply on top of these replay settings.

## TCP Target Issue: Robot Moves Away From The Goal Frame (2026-06-12)

### Symptom

When sending a TCP target from the Target tab, the robot sometimes:

- rotated the base "the wrong way" instead of moving toward the goal frame,
- followed a wild, sweeping trajectory through unexpected configurations,
- ended in a pose different from the previewed ghost.

### Root Causes Found

1. **MoveIt KDL IK returns random far-away branches.** The UR MoveIt config
   uses `kdl_kinematics_plugin/KDLKinematicsPlugin`. When the seeded attempt
   fails to converge, KDL restarts from random seeds and `/compute_ik` returns
   an arbitrary IK branch. Measured on the real robot, two identical requests
   returned `[2.31, -4.16, 1.88, 4.84, 2.56, 1.15]` and
   `[-3.97, 3.97, -1.22, -3.33, -2.56, -1.99]` for the same pose while the
   robot stood at `[-1.35, 0.20, 0.31, -2.00, -0.31, 0.00]`. Executing such a
   solution as a single joint-space segment sweeps the arm across the cell.
   Because execution re-solved IK, the executed branch could even differ from
   the previewed ghost.

2. **Euler convention mismatch in the 3D viewer.** The backend, TF, and the
   numeric RPY fields use the ROS convention (extrinsic X-Y-Z, i.e.
   `R = Rz(yaw)·Ry(pitch)·Rx(roll)`), but `viewer3d.js` rendered and read the
   target frame with three.js Euler order `"XYZ"` (intrinsic X-Y-Z). With more
   than one non-zero angle the displayed gizmo orientation did not match the
   pose actually sent to IK, so rotating the gizmo moved the robot "in the
   wrong direction".

### Fixes Applied

Backend (`ur3e_web_ui/app.py`, `ur3e_web_ui/motion.py`):

- `/compute_ik` is now sampled up to 12 times with seeds perturbed around the
  current joints (plus a long-timeout fallback); each solution is wrapped by
  whole turns (±2π, identical TCP pose) toward the current joints and the
  branch with the smallest weighted joint motion is selected. Retrying stops
  early once a solution within 0.35 rad of the seed is found. Verified: 10/10
  identical plans for a target that previously returned 4 different branches.
- The exact home pose is a wrist singularity (`wrist_2 = 0`): KDL cannot
  converge from or onto it, so IK for targets keeping the exact home
  orientation fails with a clear error instead of moving. Jog `wrist_2` a few
  degrees off zero (or tilt the target) before using the Target tab from home.
- `/api/tcp_target/plan` returns `max_joint_delta_rad`; the UI shows the
  largest planned joint move next to "IK ok".
- `/api/tcp_target/execute` accepts `expected_joints_rad` (the joints shown
  during validation) and rejects the goal with HTTP 409 if the fresh IK
  solution deviates by more than 0.05 rad on any joint. What the ghost
  previews is what the robot executes.

Frontend (`static/js/viewer3d.js`, `static/js/target_panel.js`):

- Target frame orientation now uses three.js Euler order `"ZYX"` in both
  directions, matching ROS RPY.
- The Target tab passes the validated joint target as `expected_joints_rad`
  on execute.

### Verification

With the robot live, requesting an IK plan for the *current* TCP pose returns
the current joints (≤0.02 rad difference), and repeated plans for an offset
target now return the same near-seed branch instead of random far branches.

## TCP Goals All Rejected: Controller Left Inactive (2026-06-12)

### Symptom

Every Send Goal showed `tcp: rejected` even though IK validated, the
`controller` badge said ready, `program: running`, and `speed > 0%`.

### Root Cause

`/tmp/ur3e_driver.log` showed:

```text
[scaled_joint_trajectory_controller]: Can't accept new action goals. Controller is not running.
```

The driver's `controller_stopper` deactivates motion controllers whenever the
External Control program stops (e.g. during a stack restart). In a race around
driver startup / program re-Play, `scaled_joint_trajectory_controller` can be
left **inactive** while the program reports running. The action server still
exists (so the UI badge looked ready) but rejects every goal.

Manual fix:

```bash
ros2 control switch_controllers --activate scaled_joint_trajectory_controller
```

### Fixes Applied

- The bridge now polls `/controller_manager/list_controllers` (1 Hz) and
  tracks the real `scaled_joint_trajectory_controller` state.
- Motion endpoints auto-reactivate the controller through
  `/controller_manager/switch_controller` when it is inactive while the
  program runs (activation only holds position, no motion); if reactivation
  fails they return a clear 409 instead of letting the goal be rejected.
- The `controller` badge shows `inactive` when the action server exists but
  the controller is deactivated.

## Verification Commands

Check robot network:

```bash
ip route get 192.168.0.5
ping -c 3 192.168.0.5
```

Check ROS driver health:

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic echo /joint_states --once
ros2 action info /scaled_joint_trajectory_controller/follow_joint_trajectory
ros2 control list_controllers
```

Check External Control / speed state:

```bash
ros2 topic echo /speed_scaling_state_broadcaster/speed_scaling --once
ros2 topic echo /io_and_status_controller/robot_program_running --once
```

Check UI health:

```bash
curl -fsS http://127.0.0.1:8080/api/health
curl -fsS http://127.0.0.1:8080/api/replay_settings
```

Expected UI health when ready:

```json
{"ok":true,"joint_states_alive":true,"action_server_ready":true}
```

## Current Working Procedure

1. Connect the robot Ethernet.
2. Confirm PC wired IP is `192.168.0.3`.
3. Start the stack:

   ```bash
   source script.zsh
   ur3e_stack
   ```

4. On the teach pendant, load/start the External Control program.
5. Set speed slider above `0%`, starting low for safety.
6. Open:

   ```text
   http://127.0.0.1:8080
   ```

7. Confirm badges:

   ```text
   joints: live
   controller: ready
   program: running
   speed: > 0%
   ```

8. Test small jog movement first.
9. Validate/preview replay before executing on the real robot.

## Useful Mental Model

If the robot does not move, split the problem into layers:

```text
Ping OK
  -> Ethernet route is probably OK.

Dashboard works
  -> Dashboard server is reachable, but this does not prove trajectory execution works.

/joint_states live
  -> RTDE/state streaming works, but this does not prove External Control is executing.

Action server ready
  -> ROS controller is available, but the robot may still be stopped by External Control or speed scaling.

program running + speed > 0%
  -> Real motion should be possible if the trajectory is valid and safe.
```

