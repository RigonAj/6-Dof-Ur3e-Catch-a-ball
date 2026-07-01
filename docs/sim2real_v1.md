# Sim2real V1 Workflow

This workflow turns an Isaac Lab PPO run into artifacts that can be checked,
replayed on a UR3e, and compared against the simulator before any closed-loop
real-ball command path is attempted.

## Goal

V1 is deliberately conservative:

1. export a deterministic policy and machine-readable metadata;
2. record rollout samples with simulator post-action state;
3. validate the export contract and rollout safety limits;
4. replay one rollout through the Universal Robots ROS 2 driver;
5. log `/joint_states` and compare real joints against Isaac's post-action joint
   positions.

It does not run the policy live from real perception. Live policy control also
requires exact 33-D observation reconstruction, calibrated ball and disk poses,
latency measurement, and robot-side command gating.

## Action Contract

The current environment action is an incremental joint-position target, not the
legacy absolute mapping. The policy outputs six normalized values. Isaac clips
them to `[-1, 1]`, converts them to a per-step joint delta with
`joint_velocity_safe_rad_s * dt_s`, then applies acceleration and joint-limit
clamps before integrating the previous `joint_position_target_rad`. Integrating
the previous target, instead of recomputing from measured `q` every step, gives
the position actuator a real target trajectory to follow while keeping command
velocity bounded.

For robot replay, always use `joint_position_target_rad`. Do not stream
`action_normalized` directly to the UR3e.

The metadata fields that define the contract are:

- `dt_s`
- `joint_names`
- `action_semantics`
- `action_delta_scale_rad`
- `joint_velocity_safe_rad_s`
- `joint_acceleration_safe_rad_s2`
- `joint_position_lower_rad`
- `joint_position_upper_rad`
- `rollout_schema_version`

The UR3e actuator limits used by the current Isaac config are aligned to
`ur_description`: velocity `[pi, pi, pi, 2*pi, 2*pi, 2*pi] rad/s`, effort
`[56, 56, 28, 12, 12, 12] Nm`, and initial acceleration envelope
`a_safe = 4 * velocity_limit`.

An old export that says
`joint_position_target_rad = action_normalized * action_scale` is a legacy
absolute-action export. Keep it for historical comparison, but do not treat it
as compatible with the current environment or live-catch mapper.

## Export

From the project root:

```bash
source script.zsh
sim2real_export
```

This runs `scripts/skrl/play.py` headless with one environment, exports
TorchScript and ONNX policies, writes `policy_metadata.json`, and records rollout
episodes under the selected run's `exports/` directory.

The manual equivalent is:

```bash
HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/play.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=1 \
  --checkpoint <path-to-best_agent.pt> \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
  --export_policy \
  --export_onnx \
  --record_actions \
  --record_episodes=10
```

## Validate The Export

Run both validators before copying artifacts to another machine or replaying on
hardware:

```bash
source script.zsh
sim2real_validate
```

Manual commands:

```bash
python scripts/sim2real_validate_export.py \
  --exports logs/skrl/cartpole_direct/<run>_ppo_torch/exports

python scripts/sim2real_validate_rollout_safety.py \
  --rollout logs/skrl/cartpole_direct/<run>_ppo_torch/exports/rollouts_10_episodes.json
```

`sim2real_validate_export.py` checks file presence, metadata dimensions, UR3e
joint order, rollout schema version, action semantics and rollout sample fields.
By default it rejects the legacy absolute-action metadata.

`sim2real_validate_rollout_safety.py` checks the recorded joint targets against
the position, velocity and acceleration limits stored in `policy_metadata.json`.

## Replay And Compare

Use the Universal Robots ROS 2 driver and
`scaled_joint_trajectory_controller/follow_joint_trajectory` for V1 replay. This
keeps speed scaling and the normal driver safety path in control.

While replaying, log real joint states:

```bash
python scripts/ros2_log_joint_states_csv.py \
  --output real_joint_states_episode0.csv
```

Then compare the real log with Isaac's post-action state:

```bash
python scripts/compare_sim_real_rollout.py \
  --rollout logs/skrl/cartpole_direct/<run>_ppo_torch/exports/rollouts_10_episodes.json \
  --real-log real_joint_states_episode0.csv \
  --episode 0 \
  --output sim_real_episode0_report.json
```

For sim-to-real comparison, compare against `joint_position_after_rad`, not
`joint_position_target_rad`. The target is the command; the after-state is what
Isaac's simulated robot actually reached after one policy step.

## Real-Robot Gate

Before physical replay:

- validate the export and rollout safety scripts with no failures;
- start with fake hardware or URSim;
- remove the ball and unnecessary obstacles;
- start the real UR3e in reduced speed mode;
- prepend a slow approach to the first rollout target;
- keep an operator at the E-stop;
- compare joint logs before attempting real ball interception.

RTDE/URScript `servoj` is not part of V1. It can be evaluated later, but it is
more timing-sensitive than the trajectory-controller replay path.
