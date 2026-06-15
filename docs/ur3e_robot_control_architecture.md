# UR3e Robot Control Architecture

Date: 2026-06-12

This document describes the architecture and main functions of the UR3e robot-control side of this project: ROS 2 driver launch, web UI, jog/home control, TCP target control, MoveIt IK, and rollout replay from Isaac Lab.

The Isaac Lab training environment is still part of the repository, but this document focuses on the path that controls or previews the real UR3e.

## High-Level Architecture

```text
Browser UI
  |
  | HTTP JSON API + WebSocket state stream
  v
ur3e_web_ui FastAPI backend
  |
  | rclpy node: subscriptions, TF, services, actions
  v
ROS 2 graph
  |
  | /scaled_joint_trajectory_controller/follow_joint_trajectory
  | /joint_states
  | /tf
  | /compute_ik
  | /dashboard_client/*
  v
UR ROS 2 driver + MoveIt
  |
  | reverse interface / RTDE / Dashboard
  v
Physical UR3e
```

The backend never sends Cartesian commands directly to the UR controller. Every physical move is converted into a joint trajectory and sent through the UR scaled joint trajectory controller.

## Main Runtime Processes

`scripts/launch_ur3e_stack.sh` starts the robot-control stack in order:

1. Stops stale UR driver, MoveIt, and UI processes.
2. Sources `/opt/ros/humble/setup.bash`.
3. Sources `ros2_ws/install/setup.bash`.
4. Checks that the robot at `192.168.0.5` is reachable.
5. Launches `ur_robot_driver` for `ur3e`.
6. Waits for `/joint_states` and `/scaled_joint_trajectory_controller/follow_joint_trajectory`.
7. Launches MoveIt `move_group` from `ur_moveit_config` for `/compute_ik`.
8. Launches the web UI on `http://127.0.0.1:8080`.
9. Stops UI, MoveIt, and driver cleanly on `Ctrl+C`.

Convenience aliases in `script.zsh`:

```bash
ur3e_stack
ur3e_stop
```

MoveIt can be disabled if Cartesian TCP targets are not needed:

```bash
ur3e_stack --no-moveit
```

## ROS 2 Packages In This Repo

### `ur3e_web_ui`

Path:

```text
ros2_ws/src/ur3e_web_ui
```

Purpose:

- Serves the browser UI.
- Maintains ROS state in a bridge node.
- Sends jog/home/TCP/replay trajectories.
- Exposes API endpoints and WebSocket state.
- Loads URDF for the 3D viewer.

Important files:

```text
ur3e_web_ui/app.py             FastAPI app and API endpoints
ur3e_web_ui/ros_interface.py   rclpy bridge to topics, TF, services, actions
ur3e_web_ui/motion.py          Builds small joint-space plans
ur3e_web_ui/joint_limits.py    Loads UR3e joint limits from ur_description
ur3e_web_ui/urdf_provider.py   Loads robot_description or xacro fallback
static/index.html              UI layout
static/js/main.js              App boot, WebSocket, global status
static/js/viewer3d.js          Three.js URDF viewer, ghost robot, target frame
static/js/jog_panel.js         Per-joint jog/home/cancel UI
static/js/target_panel.js      TCP target UI and IK validation
static/js/rollout_panel.js     Replay settings, preview, execution
static/css/app.css             UI styling
```

### `ur3e_rollout_replay`

Path:

```text
ros2_ws/src/ur3e_rollout_replay
```

Purpose:

- Loads Isaac rollout JSON files.
- Validates joint order, timing, and schema.
- Retimes recorded joint targets with safety limits.
- Sends replay trajectories through `FollowJointTrajectory`.

Important files:

```text
replay_core.py   Rollout schema, retiming, stats, safety limits
send.py          CLI sender and JointTrajectory builder
validate.py      CLI validation tool
```

Commands:

```bash
ros2 run ur3e_rollout_replay ur3e_replay_validate
ros2 run ur3e_rollout_replay ur3e_replay_send
```

## Browser UI Tabs

### Status

Shows:

- Joint positions in radians/degrees.
- Joint velocities.
- TCP pose from TF: `base -> tool0`.
- Driver status badges:
  - `joints`
  - `controller` (`ready` / `inactive` / `down`; `inactive` means the action
    server exists but `scaled_joint_trajectory_controller` is deactivated and
    will reject every goal)
  - `program`
  - `speed`
  - `ik`
  - `urdf`

### Control

Provides:

- Per-joint jog buttons.
- Jog step selector.
- Move Home.
- Cancel motion.
- Dashboard controls when available:
  - Play
  - Stop
  - Power On
  - Power Off
  - Brake Release

Jog and home moves use joint-space trajectories.

### Target

Provides Cartesian-style TCP targeting:

- 3D target frame in the viewer.
- Move XYZ gizmo.
- Rotate XYZ gizmo.
- Numeric fields:
  - X, Y, Z in meters
  - Roll, Pitch, Yaw in degrees
  - duration in seconds
- `Use TCP` button to copy the live TCP pose into the target.
- `Validate IK` button.
- `Send Goal` button.

Flow:

```text
User moves target in 3D or edits fields
  -> UI sends pose to /api/tcp_target/plan
  -> backend calls MoveIt /compute_ik
  -> IK joint target is retimed into a joint trajectory
  -> ghost robot previews final joint target
  -> user confirms Send Goal
  -> backend sends FollowJointTrajectory to UR driver
```

The robot does not move when the target frame is dragged. Motion happens only after validation and explicit send.

### Rollout

Provides replay of Isaac Lab rollouts:

- Lists recorded episodes from `rollouts_10_episodes.json`.
- Shows rollout metadata and retimed duration.
- Allows replay settings changes:
  - max joint velocity
  - max joint acceleration
  - approach min duration
  - min segment duration
  - Safe / Balanced / Fast presets
- Validates a replay plan.
- Previews it with a blue ghost robot for the real-robot plan and, when the
  plan includes the live approach, an amber ghost for the recorded replay
  without the approach.
- Executes after confirmation.

Replay execution always recomputes from the current physical robot joint state, so an approach segment is prepended from the current robot pose to the first recorded replay target.

## Backend API

The FastAPI backend is in `ur3e_web_ui/app.py`.

Main endpoints:

```text
GET  /                         Browser UI
GET  /api/health               Basic backend/ROS readiness
GET  /api/state                Full state snapshot for debugging
GET  /api/urdf                 URDF XML for the 3D viewer
GET  /api/limits               Joint, jog, home, safety limits

POST /api/jog                  Move one joint by a small step
POST /api/move_home            Move to configured home pose
POST /api/cancel               Cancel active motion
POST /api/dashboard/{command}  Dashboard service commands

GET  /api/replay_settings      Current replay retiming settings
POST /api/replay_settings      Update replay retiming settings
GET  /api/rollout              List rollout episodes
GET  /api/rollout/{i}/plan     Build/validate one replay plan
POST /api/rollout/{i}/execute  Execute one replay

POST /api/tcp_target/plan      Solve IK and build TCP target plan
POST /api/tcp_target/execute   Execute TCP target plan

WS   /ws                       State and goal progress stream
```

Motion endpoints enforce safety gates before sending a physical trajectory:

- `/joint_states` must be alive.
- Action server must be ready.
- External Control must be running if the topic is available.
- Speed scaling must be above `0%` if the topic is available.
- `scaled_joint_trajectory_controller` must be active; if it is inactive while
  External Control runs, the backend tries to reactivate it before sending.
- Robot joint velocity must be near zero for replay/TCP execution.
- Only one non-jog motion goal is allowed at a time.

## ROS Bridge

`ur3e_web_ui/ros_interface.py` owns one `rclpy` node running in a background executor thread.

It subscribes to:

```text
/joint_states
/robot_description
/speed_scaling_state_broadcaster/speed_scaling
/io_and_status_controller/robot_program_running
/tf
```

It calls services:

```text
/dashboard_client/play
/dashboard_client/stop
/dashboard_client/power_on
/dashboard_client/power_off
/dashboard_client/brake_release
/compute_ik
/controller_manager/list_controllers
/controller_manager/switch_controller
```

`list_controllers` is polled at about 1 Hz to track whether
`scaled_joint_trajectory_controller` is `active`. `switch_controller` is used
to reactivate it when a motion is requested while it is inactive (the UR
`controller_stopper` deactivates motion controllers when External Control
stops, and a startup/re-Play race can leave it inactive even though the action
server is up). Reactivation only re-activates the controller to hold position;
it does not move the robot.

It sends action goals to:

```text
/scaled_joint_trajectory_controller/follow_joint_trajectory
```

It maintains a thread-safe `StateSnapshot` with:

- latest joint positions/velocities
- latest TCP pose
- action server readiness
- trajectory controller active state
- IK service readiness
- dashboard availability
- speed scaling
- External Control program running state
- active goal status

## Motion Planning Model

There are two kinds of planning in this project.

### Joint-Space Retiming

Implemented in:

```text
ur3e_rollout_replay/replay_core.py
ur3e_web_ui/motion.py
```

The retimer computes a duration for each joint-space segment from:

- max joint displacement
- configured max joint velocity
- configured max joint acceleration
- minimum segment duration

It does not do collision planning. It is a conservative joint-space timing tool.

### MoveIt IK For TCP Targets

Implemented through:

```text
/compute_ik
```

The UI requests an IK solution for:

```text
group_name: ur_manipulator
ik_link_name: tool0
pose frame: base_link
```

The user-facing target fields are in the UR `base` frame. The backend/viewer must convert correctly between:

```text
base      UR controller / teach-pendant style frame
base_link REP-103 MoveIt planning frame
```

On UR descriptions, `base` is rotated `pi` radians around Z relative to `base_link`. This is why TCP target code has explicit `base <-> base_link` handling.

## Coordinate Frames

Key frames:

```text
base_link   MoveIt planning frame
base        UR controller-style base frame
tool0       TCP/end-effector frame used by the UI
```

Live TCP display:

```text
TF lookup: base -> tool0
```

MoveIt IK:

```text
IK request frame: base_link
IK tip link: tool0
```

Three.js viewer:

```text
ROS Z-up is converted into Three.js Y-up.
URDF robot is rotated by -pi/2 around X.
```

The TCP target UI is intended to be user-facing in `base`, not `base_link`.

## Data Flow: Jog

```text
Browser button press
  -> POST /api/jog
  -> app.py validates live joint state and motion gates
  -> motion.build_jog_target()
  -> RosBridge.send_plan(kind="jog")
  -> FollowJointTrajectory action
  -> UR scaled joint trajectory controller
```

Jog is special: repeated jog commands can build from the last in-flight jog target for smoother hold-to-jog behavior.

## Data Flow: Move Home

```text
Browser Move Home
  -> POST /api/move_home
  -> app.py validates motion gates
  -> motion.build_home_plan()
  -> FollowJointTrajectory action
```

Default home pose:

```text
(0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0)
```

## Data Flow: TCP Target

```text
Browser Target tab
  -> target_panel.js reads target frame / numeric fields
  -> POST /api/tcp_target/plan
  -> app.py validates request
  -> ros_interface.solve_ik()
  -> MoveIt /compute_ik
  -> motion.build_joint_target_plan()
  -> ghost robot displays final joint solution
```

Execution:

```text
Send Goal
  -> POST /api/tcp_target/execute {"confirm": true, "expected_joints_rad": [...]}
  -> solve IK again, seeded with the validated joint target
  -> reject (409) if the fresh solution deviates > 0.05 rad from the preview
  -> build joint target plan
  -> FollowJointTrajectory action
```

This means validation is useful for preview, and execution recomputes from
current state but is guaranteed to land on the previewed IK branch (or be
rejected).

IK branch selection: MoveIt's KDL solver restarts from uniform-random seeds
when its seeded attempt fails (always the case at the singular home pose,
wrist_2 = 0), which used to return arbitrary far-away IK branches and produce
wild sweeping motions. The backend now performs several short-timeout
`/compute_ik` calls seeded at the current joints plus growing perturbations,
wraps each solution by whole turns toward the current joints, and picks the
branch with the least weighted joint motion. The largest joint move is
reported as `max_joint_delta_rad` and shown in the Target tab.

## Data Flow: Rollout Replay

```text
Isaac rollout JSON
  -> ur3e_rollout_replay.load_episode_targets()
  -> build_replay_plan()
  -> optional current-position approach segment
  -> retime_segments()
  -> UI preview / FollowJointTrajectory execution
```

Rollout file:

```text
logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/rollouts_10_episodes.json
```

Each replay sample uses:

```text
joint_position_target_rad
```

These are absolute joint positions, not velocities, torques, or Cartesian commands.

## Safety And Operator Gates

The software adds practical gates, but it does not replace robot safety practice.

Before physical motion:

- External Control program must be playing.
- Speed slider must be above `0%`.
- Workspace must be clear.
- Operator must be near E-stop.
- Start with low speed.
- Preview target/replay with ghost robot first.

Software gates:

- Rejects motion when joint state is missing.
- Rejects motion when External Control is stopped.
- Rejects motion when speed scaling is `0%`.
- Reactivates `scaled_joint_trajectory_controller` if it was left inactive,
  and rejects motion if reactivation fails.
- Rejects replay/TCP if the robot is already moving.
- Retimes planned joint-space motion.
- Checks IK target joints against position limits.
- Requires confirmation for physical replay/TCP execution.

## Current Known Sharp Edges

- TCP target depends on correct `base` / `base_link` conversion (verified correct on 2026-06-12: planning IK for the live TCP pose returns the current joints).
- RPY everywhere (backend, TF, UI fields, viewer) means ROS extrinsic X-Y-Z (`R = Rz·Ry·Rx`); the three.js viewer must use Euler order `"ZYX"`, not `"XYZ"` (fixed 2026-06-12).
- MoveIt IK can fail for reachable-looking poses due to orientation, joint seed, collision checking, or solver limits. The exact home pose is a wrist singularity (wrist_2 = 0) where KDL cannot converge; the backend works around it with perturbed seeds.
- TCP target motion uses IK plus a single joint-space segment; it is not full Cartesian path planning.
- Rollout replay is open-loop recorded joint target replay, not live policy control.
- Live policy control would require reconstructing the 33-D observation on the real robot, including ball perception.
- If every motion goal is rejected while the `controller` badge looks fine, check that `scaled_joint_trajectory_controller` is actually `active` (`ros2 control list_controllers`): the UR `controller_stopper` deactivates it whenever External Control stops. The badge shows `inactive` for this case and the backend auto-reactivates it on the next motion.

## Debug Commands

Health:

```bash
curl -fsS http://127.0.0.1:8080/api/health
curl -fsS http://127.0.0.1:8080/api/state
```

ROS action:

```bash
ros2 action info /scaled_joint_trajectory_controller/follow_joint_trajectory
```

Controller state (rejected goals with a healthy-looking badge):

```bash
ros2 control list_controllers | grep scaled
ros2 control switch_controllers --activate scaled_joint_trajectory_controller
```

MoveIt IK:

```bash
ros2 service list | grep compute_ik
```

TF:

```bash
ros2 run tf2_ros tf2_echo base tool0
ros2 run tf2_ros tf2_echo base_link tool0
```

Driver logs:

```bash
tail -80 /tmp/ur3e_driver.log
tail -80 /tmp/ur3e_moveit.log
tail -80 /tmp/ur3e_web_ui.log
```

## Recommended Development Workflow

For backend/frontend code changes:

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-select ur3e_web_ui
```

Then restart:

```bash
source script.zsh
ur3e_stop
ur3e_stack
```

For frontend-only static JS/CSS changes, the symlink install usually means a browser hard refresh is enough:

```text
Ctrl+Shift+R
```

For physical motion testing:

1. Test with fake hardware or tiny moves first.
2. Use the UI ghost preview.
3. Keep speed low.
4. Move only a few centimeters when validating frame directions.
