# 6-DoF UR3e Catch a Ball

Isaac Lab reinforcement-learning project where a 6-DoF Universal Robots UR3e arm learns to catch
or intercept a moving ball with a hoop mounted on the wrist.

![UR3e catch-a-ball environment](Img/Screenshot%20from%202026-05-26%2019-34-10.png)

## Overview

This repository is an Isaac Lab extension based on the standard external-project template, customized
for a direct RL task:

- Robot: UR3e arm loaded from local USD assets in `USD_File/`.
- Task: move the wrist-mounted hoop so the ball crosses the disk trigger.
- Policy: SKRL PPO agent with Isaac Lab vectorized environments.
- Evaluation: headless play mode can run many completed episodes and report a success rate.
- Debugging: optional red 3D marker at the disk center.

The registered Gym task is:

```bash
Template-Firsttraining-Direct-v0
```

## Project Layout

```text
.
|-- Img/                         # README screenshots
|-- USD_File/                    # UR3e / hoop USD and mesh assets
|-- scripts/skrl/                # SKRL train and play scripts
|-- source/FirstTraining/
|   |-- setup.py                 # Python package metadata
|   `-- FirstTraining/tasks/direct/firsttraining/
|       |-- firsttraining_env.py
|       |-- firsttraining_env_cfg.py
|       |-- ur_gripper.py
|       `-- agents/skrl_ppo_cfg.yaml
`-- script.zsh                   # Convenience aliases
```

## Requirements

- Ubuntu with an NVIDIA GPU.
- Isaac Sim / Isaac Lab installed and working.
- Python environment used by Isaac Lab.
- SKRL and PyTorch from the Isaac Lab environment.
- TensorBoard for training curves.

If TensorBoard is missing, install this package in the Isaac Lab environment:

```bash
python -m pip install -e source/FirstTraining
```

`source/FirstTraining/setup.py` includes `tensorboard` as a dependency.

## Installation

From the repository root:

```bash
source ~/env_isaaclab/bin/activate
python -m pip install -e source/FirstTraining
```

Check that the task is visible:

```bash
python scripts/list_envs.py
```

## Training

You can train directly with:

```bash
HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/train.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=12000 \
  --headless \
  --livestream 0 \
  --rendering_mode performance
```

Or load the helper aliases:

```bash
source script.zsh
train
```

Training logs and checkpoints are written under:

```text
logs/skrl/cartpole_direct/
```

The directory name is inherited from the original template config.

## Results

The current trained policy reaches about **98% success rate** in headless evaluation with ball spawn
noise enabled at `ball_position_noise_std = 0.05`, i.e. a 5 cm Gaussian standard deviation.

## Play

To run the newest checkpoint found by `script.zsh` without typing the run name:

```bash
source script.zsh
play
```

`play` resolves the most recently modified checkpoint under `logs/skrl/cartpole_direct/`, looking at
both `best_agent.pt` and `agent_*.pt`. Useful variants:

```bash
checkpoint          # print the checkpoint that play will use
checkpoint best     # print the newest best_agent.pt, falling back to latest if none exists
play latest         # same as play
play best           # force the newest best_agent.pt
play_latest         # alias-style helper for latest
play_best           # alias-style helper for best
```

Interactive `play` opens an Isaac Sim dashboard with pause, one-step advance, simulation speed,
current action values, reward, ball/disk position and joint target error for environment 0. It defaults to
one environment for readability. Use `play --num_envs=32` for multi-env visual stress tests,
`play --disable_play_ui` to hide the dashboard, or `play --sim_speed=0.5` to start slower.

To record a short video:

```bash
source script.zsh
record
```

## Evaluation

The play script supports a headless success-rate evaluation mode:

```bash
HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/play.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=512 \
  --checkpoint <path-to-best_agent.pt> \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
  --eval_episodes=200000
```

Or with the alias:

```bash
source script.zsh
evaluate
```

The reported success rate is cumulative over all completed episodes.

## Export and Action Rollouts

To export the trained SKRL checkpoint as a deterministic inference policy and save its metadata:

```bash
HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/play.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=1 \
  --checkpoint logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/checkpoints/best_agent.pt \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
  --export_policy \
  --export_onnx
```

This writes files under:

```text
logs/skrl/cartpole_direct/2026-05-26_17-13-29_ppo_torch/exports/
```

To simulate 10 completed episodes and save every policy action:

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

The rollout JSON contains one list per completed episode. Each sample includes the 33-D observation,
the 6-D normalized policy action, the UR3e joint position target actually sent to Isaac Lab, and the
post-physics simulator state used for sim-to-real comparison.

```text
joint_position_target_rad = previous_joint_position_target_rad + bounded_delta_q
```

For the current task, `bounded_delta_q` is produced by clipping `action_normalized` to `[-1, 1]`,
scaling by `joint_velocity_safe_rad_s * dt_s`, then applying acceleration and joint-limit clamps. The
`joint_names`, `dt_s`, limits and action semantics are written to `policy_metadata.json` and copied into
the rollout metadata. Treat recorded targets as simulation commands; validate limits, timing, collision
behavior, and an emergency-stop path before sending any replay to a real UR3e.

For the recommended V1 sim-to-real workflow, including export validation and rollout safety checks, see
[Sim2real V1 Workflow](docs/sim2real_v1.md). For the real-robot replay workflow using the Universal
Robots ROS 2 driver, see [UR3e Real-Robot Replay Guide](docs/ur3e_real_robot_replay.md).
That guide also covers comparing the simulator's post-action joint positions against measured real
UR3e joint states after replaying the same episode.

## Useful Configuration

Most task parameters are in:

```text
source/FirstTraining/FirstTraining/tasks/direct/firsttraining/firsttraining_env_cfg.py
```

Useful flags and ranges:

- `ball_spawn_x_range`, `ball_spawn_y_range`, `ball_spawn_z_range`: randomized ball spawn position.
- `enable_ball_position_noise`: enable Gaussian noise on ball spawn position.
- `ball_position_noise_std`: Gaussian noise standard deviation in meters.
- `disk_radius`: trigger radius in meters. Set `<= 0` to infer it from the Disk mesh.
- `joint_velocity_safe_rad_s`: per-joint velocity envelope used to convert normalized actions into
  per-step joint deltas.
- `joint_acceleration_safe_rad_s2`: per-joint acceleration envelope applied before commanding targets.
- `UR3E_EFFORT_LIMITS_NM` in `ur_gripper.py`: effort limits aligned with `ur_description`
  `[56, 56, 28, 12, 12, 12]` Nm.
- `enable_disk_center_marker`: show a red marker at the disk center.
- `reset_on_success`: reset the episode immediately after a successful pass-through.

PPO hyperparameters are in:

```text
source/FirstTraining/FirstTraining/tasks/direct/firsttraining/agents/skrl_ppo_cfg.yaml
```

## Notes

- The disk trigger pose is read from the USD mesh at startup, relative to `wrist_3_link`.
- A pass is accepted from either direction through the disk plane.
- The environment is optimized for headless training with cameras and livestream disabled.
- The USD assets in `USD_File/` are part of this project and should stay in the repository.
