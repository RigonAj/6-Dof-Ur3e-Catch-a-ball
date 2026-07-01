# UR3e Real-Robot Replay Guide

This guide explains how to take the rollout actions exported from the Isaac Lab policy and replay them on a real UR3e using the Universal Robots ROS 2 driver. For the full conservative export/validation workflow, run through [Sim2real V1 Workflow](sim2real_v1.md) first.

The exported rollout file is:

```text
logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/rollouts_10_episodes.json
```

The model/export metadata is:

```text
logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/policy_metadata.json
```

If you want to compare simulation against the real robot, regenerate the rollout with the current
`scripts/skrl/play.py` before collecting real data. New rollouts include the simulator's actual
post-action joint state, which older exported JSON files do not contain.

## What the Actions Mean

Each rollout sample contains the policy output, the command target sent to Isaac Lab, and the simulator state after that command:

- `action_normalized`: raw policy output from the SKRL agent.
- `joint_position_target_rad`: the command that was sent to the Isaac Lab articulation.
- `joint_position_after_rad`: where the simulator actually was after applying that command for one step.
- `joint_position_target_error_after_rad`: target minus simulated post-step position.

For the real robot, use `joint_position_target_rad`. Do not stream `action_normalized` directly to the UR3e.

In the current task, `action_normalized` is not an absolute joint target. Isaac clips it to `[-1, 1]`, scales it by the per-joint `joint_velocity_safe_rad_s * dt_s`, then applies acceleration and joint-limit clamps before integrating the previous target into a new absolute target:

```text
joint_position_target_rad = previous_joint_position_target_rad + bounded_delta_q
dt_s = 0.016666666666666666
```

The old metadata string `joint_position_target_rad = action_normalized * action_scale` describes legacy exports only. Regenerate exports with the current `scripts/skrl/play.py` before using them as the V1 sim-to-real source of truth.

The 6 joint targets are absolute joint positions in radians, ordered as:

```text
shoulder_pan_joint
shoulder_lift_joint
elbow_joint
wrist_1_joint
wrist_2_joint
wrist_3_joint
```

These are not velocity commands, torque commands, or Cartesian TCP commands.

For sim-to-real comparison, compare the real robot's measured joint positions against
`joint_position_after_rad`, not against `joint_position_target_rad`. The target tells you what was
commanded; the after-state tells you where Isaac's simulated robot actually ended up.

## Recommended ROS 2 Replay Path

Use the Universal Robots ROS 2 driver and the default `scaled_joint_trajectory_controller`. The controller accepts joint trajectories and applies UR speed scaling, so it is the first path to use before any lower-level streaming approach.

Official references:

- Universal Robots ROS 2 driver overview: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/
- UR ROS 2 controllers: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/usage/controllers.html

High-level flow:

1. Start with fake hardware or URSim.
2. Load `rollouts_10_episodes.json`.
3. Select one episode.
4. Read each sample's `joint_position_target_rad`.
5. Build a `trajectory_msgs/JointTrajectory`.
6. Send it through `control_msgs/action/FollowJointTrajectory` to:

```text
/scaled_joint_trajectory_controller/follow_joint_trajectory
```

7. Only after simulation validation, run on the physical UR3e in reduced speed mode with no ball and an operator at the E-stop.

## Robot Setup Checklist

Before commanding the real robot:

- Install and configure the Universal Robots ROS 2 driver for `ur3e`.
- Install the External Control URCap on the robot if your driver setup requires it.
- Confirm the robot calibration used by the ROS 2 driver matches the physical arm.
- Set the correct TCP and payload for the hoop/end-effector.
- Remove the ball and any unnecessary obstacles for the first tests.
- Use reduced mode or a low teach-pendant speed slider.
- Keep an operator at the E-stop.
- Verify the workspace is clear for the full planned motion.
- Confirm the robot starts near the first rollout target before replay.

## Start the Driver

Use the exact launch options for your ROS 2 distribution and driver version. A typical real-robot launch looks like:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=<ROBOT_IP> \
  launch_rviz:=true
```

For dry runs, use fake hardware or URSim first. Depending on your installed driver version, fake hardware is typically launched with an option like:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=<ROBOT_IP_OR_DUMMY_IP> \
  use_fake_hardware:=true \
  launch_rviz:=true
```

Check that the scaled trajectory controller is available:

```bash
ros2 control list_controllers
```

The active motion controller should include:

```text
scaled_joint_trajectory_controller
```

## Convert a Rollout to a Joint Trajectory

A replay program should create a `FollowJointTrajectory.Goal` with:

- `trajectory.joint_names`: the six UR3e joint names listed above.
- `trajectory.points[i].positions`: `samples[i]["joint_position_target_rad"]`.
- `trajectory.points[i].time_from_start`: cumulative time from the rollout, using `metadata["dt_s"]`.

Do not immediately start the policy motion from an arbitrary real robot pose. First prepend a slow approach segment from the current measured joint state to the first rollout target.

Recommended first-pass timing:

- Approach duration: 5 to 10 seconds.
- Replay timestep: use the rollout `dt_s` only after the approach segment.
- First physical test: slow the whole trajectory down by 3x to 5x.

Minimal replay logic:

```python
import json

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

with open("logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/rollouts_10_episodes.json") as file:
    rollout = json.load(file)

dt = rollout["metadata"]["dt_s"]
episode = rollout["episodes"][0]
positions = [sample["joint_position_target_rad"] for sample in episode["samples"]]
```

The full ROS 2 node should then send those positions as a `control_msgs/action/FollowJointTrajectory` goal to `/scaled_joint_trajectory_controller/follow_joint_trajectory`.

## Record the Simulator Reference

Regenerate the rollout after the recorder update so each sample contains the simulator's post-action
state:

```bash
HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/play.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=1 \
  --checkpoint logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/checkpoints/best_agent.pt \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
  --record_actions \
  --record_episodes=10
```

The comparison fields are:

- `sim_time_after_s`: simulator time after the action has been applied.
- `joint_position_after_rad`: simulated joint positions after the action.
- `joint_velocity_after_rad_s`: simulated joint velocities after the action.
- `disk_position_after_world_m`: simulated hoop/disk center in world coordinates.
- `ball_position_after_world_m`: simulated ball position in world coordinates.

The disk and ball fields are useful for debugging the simulated episode, but a real robot comparison
needs external perception or motion capture to measure equivalent real-world positions. Without that,
compare joint positions first.

## Record the Real Robot Response

While replaying an episode on the real robot, log measured joint states from ROS 2:

```bash
python scripts/ros2_log_joint_states_csv.py \
  --output real_joint_states_episode0.csv
```

Start this logger before sending the replay trajectory, then stop it after the motion finishes. The CSV
format is intentionally simple:

```text
time_s,shoulder_pan_joint,shoulder_lift_joint,elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint
0.000000000, ...
```

If the CSV includes the slow approach segment, note the `time_s` at which the actual rollout segment
starts. Pass that value as `--real-start-time` during comparison.

## Compare Sim and Real

Use the comparison script after you have a regenerated rollout and a real joint-state CSV:

```bash
python scripts/compare_sim_real_rollout.py \
  --rollout logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/rollouts_10_episodes.json \
  --real-log real_joint_states_episode0.csv \
  --episode 0 \
  --output sim_real_episode0_report.json
```

If the real log contains the approach phase before the rollout starts:

```bash
python scripts/compare_sim_real_rollout.py \
  --real-log real_joint_states_episode0.csv \
  --episode 0 \
  --real-start-time <ROLLOUT_START_TIME_IN_REAL_CSV>
```

The report prints:

- final absolute joint error in radians and degrees
- overall RMS joint error
- maximum absolute joint error
- per-joint RMS error
- time alignment error when timestamps are available

For the first real test, focus on final joint error and maximum joint error. If those are already large
with no ball and slow speed, the sim-to-real gap is in the robot dynamics, timing, calibration, or
trajectory execution before the catch task itself.

## Safety Validation Before Real Motion

Run these checks before sending any trajectory to the physical arm:

- Joint limits: every position is inside the UR3e joint limits.
- Step size: adjacent joint targets are small enough for the replay timing.
- Velocity: estimated joint velocity between samples is acceptable.
- Acceleration: estimated acceleration is not aggressive.
- Start pose: current robot joints are close to the planned approach start.
- Collision clearance: the hoop, wrist, arm, table, and nearby objects are clear.
- End behavior: the robot has a controlled final hold or slow stop.

For this exported rollout, the 10 saved episodes contain short successful snippets, roughly 10 to 16 policy steps each. That means the real motion segment is short; most real-robot risk will come from the approach to the first target and from sim-to-real mismatch.

## Why Live Policy Control Is Different

The saved rollout is replay-only. Running the policy live on the UR3e is a separate project because the policy observation includes simulated values:

- current joint position
- current joint velocity
- disk position
- ball position
- ball-to-disk direction
- ball distance
- ball velocity
- previous signed disk crossing state
- previous action
- pass-through count

On the real robot, the joint state can come from ROS 2, but the ball and disk state require calibrated perception and frame transforms. Do not assume the exported ONNX or TorchScript policy can be connected directly to the robot without reconstructing the exact 33-D observation.

## Advanced Alternative: RTDE and `servoj`

RTDE and URScript `servoj` can be used later for lower-level joint servoing, but this is timing-sensitive and easier to destabilize. UR documents RTDE as a real-time data exchange interface, and `servoj` as online joint-position control.

Official references:

- UR RTDE guide: https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html
- UR `servoj` command: https://www.universal-robots.com/articles/ur/programming/servoj-command/

If you later use this route, stream `joint_position_target_rad` as the joint position target `q`, not `action_normalized`, and start with conservative `lookahead_time`, gain, and reduced speed settings.
