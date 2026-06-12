# UR3e Web Control UI

A browser UI for the UR3e served by the `ur3e_web_ui` ROS 2 package:

- Live 3D model of the robot (URDF + meshes from `ur_description`), driven by `/joint_states`.
- Pose information: joint positions (rad/deg), joint velocities, and TCP pose (`base` → `tool0` via TF, matching the teach-pendant convention).
- Per-joint jog buttons, move-to-home, and a cancel button. All motion goes through `/scaled_joint_trajectory_controller/follow_joint_trajectory` with positions clamped to the UR3e joint limits.
- TCP Target tab: movable 3D target frame plus numeric X/Y/Z/Roll/Pitch/Yaw fields; validates with MoveIt `/compute_ik`, previews the IK solution as a ghost robot, then sends a retimed joint trajectory after explicit confirmation.
- Rollout tab: lists the 10 exported Isaac episodes, validates the retimed plan (same math as `ur3e_replay_validate`), previews the trajectory as a semi-transparent ghost robot (no robot motion), and executes it after explicit confirmation.
- Dashboard buttons (play/stop/power on/off/brake release) appear automatically when the real driver's dashboard client is available.

Troubleshooting history for the original real-robot motion issue is documented in `docs/ur3e_motion_issue_resolution.md`.

Architecture and function-level documentation for the robot-control stack is in `docs/ur3e_robot_control_architecture.md`.

## One-Time Setup

```bash
python3 -m pip install --user fastapi "uvicorn[standard]"
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

The frontend loads `three.js` and `urdf-loader` from unpkg.com, so the PC needs internet access in the browser. The robot network (Ethernet) and internet (Wi-Fi) can coexist; the UI server itself binds to `127.0.0.1` by default.

## One-Command Real-Robot Launcher

Source the repo aliases once in a terminal:

```bash
source script.zsh
```

Then launch the real driver and web UI together:

```bash
ur3e_stack
```

The launcher stops stale UR3e driver/UI processes first, starts the ROS driver, waits for `/joint_states` and `/scaled_joint_trajectory_controller/follow_joint_trajectory`, then starts the UI at http://127.0.0.1:8080. Press `Ctrl+C` in that terminal to stop the UI and driver cleanly.

By default the launcher also starts MoveIt `move_group` so the TCP Target tab can use `/compute_ik`. Disable it only if you do not need Cartesian targets:

```bash
ur3e_stack --no-moveit
```

Replay timing can be tuned from the launcher. Defaults are conservative:

```bash
UR3E_MAX_JOINT_VELOCITY=0.5 \
UR3E_MAX_JOINT_ACCELERATION=1.0 \
UR3E_APPROACH_MIN_DURATION=3.0 \
UR3E_MIN_SEGMENT_DURATION=0.1 \
ur3e_stack
```

The UI still executes rollouts from the measured current robot pose by prepending an approach segment to the first recorded rollout target.

The same replay limits can also be changed while the server is running from the Rollout tab:

- Max velocity / max acceleration: retiming limits for the replay trajectory.
- Approach min: minimum duration for the move from the robot's current pose to the first recorded replay pose.
- Segment min: minimum duration for each recorded replay segment.
- Safe / Balanced / Fast presets: quick starting points; each plan is still recomputed and checked before execution.

The teach-pendant speed slider and UR controller limits still apply on top of these UI settings.

To only stop the stack:

```bash
ur3e_stop
```

## Run Against Mock Hardware (no robot needed)

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
ros2 run ur3e_web_ui ur3e_web_ui
```

Open http://127.0.0.1:8080 in a browser. The `urdf` badge should say `topic` (model taken from the running driver) and the `joints` badge should say `live`.

Things to try safely with mock hardware:

1. Jog each joint with the +/- buttons (hold for continuous motion).
2. Move Home, then press CANCEL MOTION mid-move.
3. In the Rollout tab: Validate episode 0, Preview it (only the blue ghost moves), then Execute it.

## Run Against the Real Robot

Follow the gates in `docs/ur3e_current_driver_setup.md` first (calibration extracted, External Control playing, reduced speed, operator at E-stop). Then:

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run ur3e_web_ui ur3e_web_ui
```

On the real robot the UI additionally shows:

- `program` badge: whether the External Control program is running on the pendant. If it says `stopped`, motion goals will be accepted but the arm will not move — press Play on the pendant (or use the Dashboard Play button in the Control tab).
- `speed` badge: the pendant speed-slider scaling.
- `ik` badge: whether MoveIt `/compute_ik` is available for TCP target validation.

Execution is double-gated: replay execution opens a modal that recomputes the plan from the robot's current pose and requires a checkbox confirmation; TCP target execution requires a fresh IK validation and browser confirmation.

## Options

```bash
ros2 run ur3e_web_ui ur3e_web_ui --help
```

- `--port 8080`, `--host 127.0.0.1` (use `--host 0.0.0.0` to reach the UI from another device — only on a trusted network; there is no authentication).
- `--rollout <path>`: a different rollouts JSON export.
- `--max-joint-velocity 0.25 --max-joint-acceleration 0.5 --approach-min-duration 10.0 --min-segment-duration 0.5`: the same conservative safety limits used by `ur3e_replay_send`.
- `--home-joints 0,-1.5708,0,-1.5708,0,0`: home pose used by the Move Home button.

## API

The backend also exposes a JSON API (interactive docs at `/docs`):

- `GET /api/health`, `GET /api/urdf`, `GET /api/limits`
- `GET /api/replay_settings`, `POST /api/replay_settings`
- `POST /api/tcp_target/plan`, `POST /api/tcp_target/execute`
- `GET /api/rollout`, `GET /api/rollout/{i}/plan?approach=true`
- `POST /api/jog {"joint": "...", "direction": 1, "step_rad": 0.05}`
- `POST /api/move_home {"confirm": true}`, `POST /api/cancel`
- `POST /api/rollout/{i}/execute {"confirm": true}`
- `POST /api/dashboard/{play|stop|power_on|power_off|brake_release}`
- `WS /ws`: state stream at 15 Hz (joints, TCP pose, goal progress, driver status)

## Notes / Limitations

- The 3D model uses the URDF published by the running driver, so it includes the extracted kinematics calibration. Without a driver it falls back to the default `ur3e` xacro (badge shows `xacro`).
- Replaying the policy live (closed-loop) is out of scope; the Rollout tab replays recorded `joint_position_target_rad` sequences only (see `docs/ur3e_real_robot_replay.md`).
- TCP target motion uses MoveIt only for inverse kinematics; the UI still sends the final motion through the UR scaled joint trajectory controller.
- One motion goal at a time; jog requests are rejected while a rollout/home/TCP goal is active.
