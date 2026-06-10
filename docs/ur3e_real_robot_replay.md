# UR3e Real-Robot Replay Guide

This guide explains how to take the rollout actions exported from the Isaac Lab policy and replay them on a real UR3e using the Universal Robots ROS 2 driver.

The exported rollout file is:

```text
logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/rollouts_10_episodes.json
```

The model/export metadata is:

```text
logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/policy_metadata.json
```

## What the Actions Mean

Each rollout sample contains two action fields:

- `action_normalized`: raw policy output from the SKRL agent.
- `joint_position_target_rad`: the command that was sent to the Isaac Lab articulation.

For the real robot, use `joint_position_target_rad`. Do not stream `action_normalized` directly to the UR3e.

In this task:

```text
joint_position_target_rad = action_normalized * 0.5
dt_s = 0.016666666666666666
```

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
