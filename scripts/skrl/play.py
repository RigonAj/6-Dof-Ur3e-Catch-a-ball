# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--sim_speed",
    type=float,
    default=1.0,
    help="Interactive play speed multiplier. Used by the Isaac Sim dashboard when enabled.",
)
parser.add_argument(
    "--disable_play_ui",
    action="store_true",
    default=False,
    help="Disable the Isaac Sim play dashboard in non-headless play mode.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=0,
    help="Run headless evaluation for N completed episodes and print the success rate.",
)
parser.add_argument(
    "--export_policy",
    action="store_true",
    default=False,
    help="Export the loaded deterministic policy as TorchScript. Use with --export_onnx for ONNX too.",
)
parser.add_argument(
    "--export_onnx",
    action="store_true",
    default=False,
    help="Also export the loaded deterministic policy as ONNX. Requires --export_policy.",
)
parser.add_argument(
    "--export_dir",
    type=str,
    default=None,
    help="Directory for exported model files. Defaults to <run_dir>/exports.",
)
parser.add_argument(
    "--record_actions",
    action="store_true",
    default=False,
    help="Run completed episodes and save per-step policy actions for replay/robot-side testing.",
)
parser.add_argument(
    "--record_episodes",
    type=int,
    default=10,
    help="Number of completed episodes to save when --record_actions is used.",
)
parser.add_argument(
    "--record_actions_path",
    type=str,
    default=None,
    help="Output JSON path for recorded actions. Defaults to <run_dir>/exports/rollouts_<N>_episodes.json.",
)
parser.add_argument(
    "--record_max_steps",
    type=int,
    default=0,
    help="Optional safety cap on total simulator steps while recording actions. 0 means no extra cap.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.export_onnx:
    args_cli.export_policy = True
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.eval_episodes > 0:
    args_cli.video = False
    args_cli.headless = True
    args_cli.livestream = 0
    args_cli.enable_cameras = False
if args_cli.record_actions:
    args_cli.video = False
    args_cli.headless = True
    args_cli.livestream = 0
    args_cli.enable_cameras = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import FirstTraining.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _select_deterministic_actions(agent, env, obs):
    outputs = agent.act(obs, None, timestep=0, timesteps=0)
    if hasattr(env, "possible_agents"):
        return {
            agent_id: outputs[-1][agent_id].get("mean_actions", outputs[0][agent_id])
            for agent_id in env.possible_agents
        }
    return outputs[-1].get("mean_actions", outputs[0])


def _tensor_row_to_list(tensor: torch.Tensor, row: int) -> list[float]:
    return tensor[row].detach().cpu().tolist()


def _target_error_to_list(target: torch.Tensor, actual: torch.Tensor, row: int) -> list[float]:
    return (target[row] - actual[row]).detach().cpu().tolist()


def _space_size(space_or_size) -> int:
    shape = getattr(space_or_size, "shape", None)
    if shape is not None:
        size = 1
        for dimension in shape:
            size *= int(dimension)
        return size
    return int(space_or_size)


def _success_buffer(base_env, extras):
    success = getattr(base_env, "_last_done_success", None)
    if success is None and isinstance(extras, dict):
        success = extras.get("success", None)
    return success


def _tensor_first_row(tensor, default=None) -> list[float] | None:
    if tensor is None:
        return default
    try:
        return tensor[0].detach().cpu().tolist()
    except Exception:
        return default


def _max_abs_delta(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b, strict=False))


class _PlayDashboard:
    """Small Isaac Sim overlay for interactive play diagnostics.

    This deliberately avoids TensorBoard or matplotlib so it can run inside the
    Omniverse UI. If omni.ui is unavailable, the caller simply runs without it.
    """

    def __init__(self, *, speed: float) -> None:
        import omni.ui as ui

        self.ui = ui
        self.paused = False
        self._step_once = False
        self._labels: dict[str, object] = {}
        self._action_labels: list[object] = []
        self._speed_model = ui.SimpleFloatModel(max(0.0, float(speed)))

        self._window = ui.Window("UR3e Play Dashboard", width=520, height=720)
        with self._window.frame:
            with ui.VStack(spacing=6):
                ui.Label("UR3e Play Dashboard", height=24)
                with ui.HStack(height=28, spacing=6):
                    self._pause_button = ui.Button("Pause", clicked_fn=self.toggle_pause)
                    ui.Button("Step", clicked_fn=self.step_once)
                ui.Label("Simulation speed", height=20)
                with ui.HStack(height=24, spacing=6):
                    ui.FloatSlider(model=self._speed_model, min=0.0, max=4.0)
                    self._labels["speed"] = ui.Label("1.00x", width=70)

                ui.Separator(height=8)
                for key in (
                    "step",
                    "reward",
                    "done",
                    "ball_pos",
                    "ball_vel",
                    "disk_pos",
                    "distance",
                    "joint_error",
                ):
                    self._labels[key] = ui.Label(f"{key}: -", height=20)

                ui.Separator(height=8)
                ui.Label("Actions [-1, 1]", height=22)
                for index in range(6):
                    label = ui.Label(f"a{index}: -", height=22)
                    self._action_labels.append(label)

    @property
    def speed(self) -> float:
        try:
            return max(0.0, float(self._speed_model.get_value_as_float()))
        except Exception:
            return 1.0

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        try:
            self._pause_button.text = "Resume" if self.paused else "Pause"
        except Exception:
            pass

    def step_once(self) -> None:
        self._step_once = True

    def should_step(self) -> bool:
        if self._step_once:
            self._step_once = False
            return True
        if not self.paused:
            return self.speed > 0.0
        return False

    def _set_label(self, key: str, text: str) -> None:
        label = self._labels.get(key)
        if label is not None:
            try:
                label.text = text
            except Exception:
                pass

    def update(self, state: dict) -> None:
        speed = self.speed
        self._set_label("speed", f"{speed:.2f}x")
        self._set_label("step", f"step: {state.get('step', 0)}  sim_t: {state.get('sim_time_s', 0.0):.3f}s")
        self._set_label("reward", f"reward: {state.get('reward', 0.0): .3f}")
        self._set_label("done", f"done: {state.get('done', False)}  success: {state.get('success', None)}")

        ball_pos = state.get("ball_pos")
        ball_vel = state.get("ball_vel")
        disk_pos = state.get("disk_pos")
        joint_error = state.get("joint_error")
        distance = state.get("distance")
        self._set_label("ball_pos", "ball pos: " + _format_vec(ball_pos, "m"))
        self._set_label("ball_vel", "ball vel: " + _format_vec(ball_vel, "m/s"))
        self._set_label("disk_pos", "disk pos: " + _format_vec(disk_pos, "m"))
        self._set_label("distance", "ball-disk distance: " + ("-" if distance is None else f"{distance:.3f} m"))
        self._set_label("joint_error", "max |target - q|: " + ("-" if joint_error is None else f"{joint_error:.3f} rad"))

        actions = state.get("actions") or []
        for index, label in enumerate(self._action_labels):
            current = actions[index] if index < len(actions) else 0.0
            try:
                label.text = f"a{index}: {current:+.3f}"
            except Exception:
                pass


def _format_vec(values, unit: str) -> str:
    if values is None:
        return "-"
    try:
        return "[" + ", ".join(f"{float(value):+.3f}" for value in values[:3]) + f"] {unit}"
    except Exception:
        return "-"


def _play_dashboard_state(base_env, actions, rewards, terminated, truncated, extras, step: int, dt: float) -> dict:
    action_values = _tensor_first_row(actions, []) if not isinstance(actions, dict) else []
    reward_value = 0.0
    try:
        reward_value = float(rewards.view(-1)[0].detach().cpu().item())
    except Exception:
        pass

    done = False
    try:
        done = bool((terminated | truncated).view(-1)[0].detach().cpu().item())
    except Exception:
        pass

    success_value = None
    success = _success_buffer(base_env, extras)
    try:
        if success is not None:
            success_value = bool(success.view(-1)[0].detach().cpu().item())
    except Exception:
        success_value = None

    ball_pos = _tensor_first_row(getattr(base_env, "_last_step_ball_pos_local", None))
    if ball_pos is None:
        ball_pos = _tensor_first_row(getattr(base_env, "_ball_pos_local", None))
    if ball_pos is None and hasattr(base_env, "ball"):
        ball_pos = _tensor_first_row(getattr(base_env.ball.data, "root_pos_w", None))

    ball_vel = _tensor_first_row(getattr(base_env, "_last_step_ball_vel_w", None))
    if ball_vel is None:
        ball_vel = _tensor_first_row(getattr(base_env, "_ball_vel_w", None))
    if ball_vel is None and hasattr(base_env, "ball"):
        ball_vel = _tensor_first_row(getattr(base_env.ball.data, "root_lin_vel_w", None))

    disk_pos = _tensor_first_row(getattr(base_env, "_last_step_disk_pos_local", None))
    if disk_pos is None:
        disk_pos = _tensor_first_row(getattr(base_env, "_disk_pos_local", None))

    distance = None
    if ball_pos is not None and disk_pos is not None:
        distance = sum((float(ball_pos[index]) - float(disk_pos[index])) ** 2 for index in range(3)) ** 0.5

    joint_pos = _tensor_first_row(getattr(base_env, "_last_step_joint_pos", None))
    if joint_pos is None and hasattr(base_env, "robot"):
        try:
            joint_pos = _tensor_first_row(base_env.robot.data.joint_pos[:, base_env._arm_dof_idx])
        except Exception:
            joint_pos = None
    joint_target = _tensor_first_row(getattr(base_env, "_last_step_joint_pos_target", None))
    if joint_target is None:
        joint_target = _tensor_first_row(getattr(base_env, "joint_pos_target", None))

    return {
        "step": step,
        "sim_time_s": step * dt,
        "reward": reward_value,
        "done": done,
        "success": success_value,
        "actions": action_values,
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "disk_pos": disk_pos,
        "distance": distance,
        "joint_error": _max_abs_delta(joint_target, joint_pos),
    }


class _DeterministicSkrlPolicy(torch.nn.Module):
    """Thin trace wrapper around SKRL's eval-time deterministic action path."""

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

        # Register known torch modules so tracing/ONNX export can see parameters owned by SKRL.
        models = getattr(agent, "models", {})
        if isinstance(models, dict):
            self._skrl_models = torch.nn.ModuleDict(
                {name: model for name, model in models.items() if isinstance(model, torch.nn.Module)}
            )
        for name in (
            "state_preprocessor",
            "_state_preprocessor",
            "observation_preprocessor",
            "_observation_preprocessor",
        ):
            module = getattr(agent, name, None)
            if isinstance(module, torch.nn.Module):
                self.add_module(f"_skrl_{name}", module)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        outputs = self.agent.act(observations, None, timestep=0, timesteps=0)
        return outputs[-1].get("mean_actions", outputs[0])


def _export_policy(agent, sample_obs: torch.Tensor, export_dir: str, export_onnx: bool, metadata: dict) -> None:
    os.makedirs(export_dir, exist_ok=True)
    wrapper = _DeterministicSkrlPolicy(agent).eval()
    sample_obs = sample_obs.detach()
    torchscript_path = os.path.join(export_dir, "policy_deterministic.ts")
    metadata_path = os.path.join(export_dir, "policy_metadata.json")

    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, sample_obs, strict=False, check_trace=False)
        traced.save(torchscript_path)
    print(f"[EXPORT] TorchScript deterministic policy: {torchscript_path}")

    if export_onnx:
        onnx_path = os.path.join(export_dir, "policy_deterministic.onnx")
        torch.onnx.export(
            wrapper,
            sample_obs,
            onnx_path,
            input_names=["observations"],
            output_names=["actions"],
            dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17,
        )
        print(f"[EXPORT] ONNX deterministic policy: {onnx_path}")

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"[EXPORT] Policy metadata: {metadata_path}")


def _record_action_rollouts(
    agent,
    env,
    base_env,
    obs,
    output_path: str,
    episodes_to_record: int,
    action_scale: float,
    dt: float,
    metadata: dict,
    max_steps: int,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    num_envs = int(getattr(env, "num_envs", base_env.num_envs))
    buffers = [[] for _ in range(num_envs)]
    episode_step = [0 for _ in range(num_envs)]
    episodes = []
    total_steps = 0
    last_reported_episodes = 0

    print(f"[RECORD] Recording {episodes_to_record} completed episodes to: {output_path}")
    while simulation_app.is_running() and len(episodes) < episodes_to_record:
        if max_steps > 0 and total_steps >= max_steps:
            raise RuntimeError(
                f"Stopped after --record_max_steps={max_steps} with only {len(episodes)} completed episodes."
            )

        with torch.inference_mode():
            actions = _select_deterministic_actions(agent, env, obs)
            if isinstance(actions, dict):
                raise RuntimeError("Action recording currently expects a single-agent tensor action space.")

            joint_pos = base_env.robot.data.joint_pos[:, base_env._arm_dof_idx]
            joint_vel = base_env.robot.data.joint_vel[:, base_env._arm_dof_idx]

            step_records = []
            for env_id in range(num_envs):
                record = {
                    "step": episode_step[env_id],
                    "time_s": episode_step[env_id] * dt,
                    "observation": _tensor_row_to_list(obs, env_id),
                    "action_normalized": _tensor_row_to_list(actions, env_id),
                    "joint_position_before_rad": _tensor_row_to_list(joint_pos, env_id),
                    "joint_velocity_before_rad_s": _tensor_row_to_list(joint_vel, env_id),
                }
                buffers[env_id].append(record)
                step_records.append(record)

            obs, rewards, terminated, truncated, extras = env.step(actions)
            if hasattr(base_env, "_last_step_joint_pos"):
                joint_pos_after = base_env._last_step_joint_pos
                joint_vel_after = base_env._last_step_joint_vel
                disk_pos_after_w = base_env._last_step_disk_pos_w
                disk_pos_after_local = base_env._last_step_disk_pos_local
                ball_pos_after_w = base_env._last_step_ball_pos_w
                ball_pos_after_local = base_env._last_step_ball_pos_local
                ball_vel_after_w = base_env._last_step_ball_vel_w
                joint_targets = base_env._last_step_joint_pos_target
                last_step_valid = base_env._last_step_valid
                sim_state_after_source = "post_physics_pre_reset_cache"
            else:
                joint_pos_after = base_env.robot.data.joint_pos[:, base_env._arm_dof_idx]
                joint_vel_after = base_env.robot.data.joint_vel[:, base_env._arm_dof_idx]
                joint_targets = getattr(base_env, "joint_pos_target", actions * action_scale)
                disk_pos_after_w = getattr(base_env, "_disk_pos_w", torch.zeros(num_envs, 3, device=actions.device))
                disk_pos_after_local = disk_pos_after_w - base_env.scene.env_origins
                ball_pos_after_w = getattr(base_env, "_ball_pos_w", torch.zeros(num_envs, 3, device=actions.device))
                ball_pos_after_local = ball_pos_after_w - base_env.scene.env_origins
                ball_vel_after_w = getattr(base_env, "_ball_vel_w", torch.zeros(num_envs, 3, device=actions.device))
                last_step_valid = torch.ones(num_envs, dtype=torch.bool, device=actions.device)
                sim_state_after_source = "articulation_after_step"

        done = (terminated | truncated).view(-1)
        success = _success_buffer(base_env, extras)
        for env_id, record in enumerate(step_records):
            record["sim_time_after_s"] = (record["step"] + 1) * dt
            record["joint_position_target_rad"] = _tensor_row_to_list(joint_targets, env_id)
            record["joint_position_after_rad"] = _tensor_row_to_list(joint_pos_after, env_id)
            record["joint_velocity_after_rad_s"] = _tensor_row_to_list(joint_vel_after, env_id)
            record["joint_position_target_error_after_rad"] = _target_error_to_list(
                joint_targets,
                joint_pos_after,
                env_id,
            )
            record["disk_position_after_world_m"] = _tensor_row_to_list(disk_pos_after_w, env_id)
            record["disk_position_after_local_m"] = _tensor_row_to_list(disk_pos_after_local, env_id)
            record["ball_position_after_world_m"] = _tensor_row_to_list(ball_pos_after_w, env_id)
            record["ball_position_after_local_m"] = _tensor_row_to_list(ball_pos_after_local, env_id)
            record["ball_velocity_after_world_m_s"] = _tensor_row_to_list(ball_vel_after_w, env_id)
            record["sim_state_after_source"] = sim_state_after_source
            record["sim_state_after_valid"] = bool(last_step_valid[env_id].detach().cpu().item())
            record["reward"] = float(rewards.view(-1)[env_id].detach().cpu().item())
            record["terminated"] = bool(terminated.view(-1)[env_id].detach().cpu().item())
            record["truncated"] = bool(truncated.view(-1)[env_id].detach().cpu().item())

            if bool(done[env_id].detach().cpu().item()):
                episode = {
                    "episode_index": len(episodes),
                    "source_env_index": env_id,
                    "success": bool(success[env_id].detach().cpu().item()) if success is not None else None,
                    "steps": len(buffers[env_id]),
                    "samples": buffers[env_id],
                }
                episodes.append(episode)
                buffers[env_id] = []
                episode_step[env_id] = 0
                if len(episodes) >= episodes_to_record:
                    break
            else:
                episode_step[env_id] += 1

        total_steps += 1
        if len(episodes) % 5 == 0 and len(episodes) > 0 and len(episodes) != last_reported_episodes:
            last_reported_episodes = len(episodes)
            print(f"[RECORD] completed episodes={len(episodes)}/{episodes_to_record}")

    payload = {
        "metadata": metadata,
        "episodes": episodes[:episodes_to_record],
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"[RECORD] Saved {len(payload['episodes'])} episodes to: {output_path}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    if args_cli.record_actions and args_cli.num_envs is None:
        env_cfg.scene.num_envs = 1
    else:
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
        base_env = env.unwrapped

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    # reset environment
    obs, _ = env.reset()

    export_dir = os.path.abspath(args_cli.export_dir) if args_cli.export_dir else os.path.join(log_dir, "exports")
    joint_names = list(getattr(base_env.cfg, "joint_names", []))
    action_scale = float(getattr(base_env.cfg, "action_scale", 1.0))
    joint_velocity_safe = list(getattr(base_env.cfg, "joint_velocity_safe_rad_s", []))
    joint_acceleration_safe = list(getattr(base_env.cfg, "joint_acceleration_safe_rad_s2", []))
    joint_position_lower = list(getattr(base_env.cfg, "joint_position_lower_rad", []))
    joint_position_upper = list(getattr(base_env.cfg, "joint_position_upper_rad", []))
    metadata = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "log_dir": log_dir,
        "ml_framework": args_cli.ml_framework,
        "algorithm": algorithm,
        "dt_s": float(dt),
        "num_envs": int(getattr(env, "num_envs", base_env.num_envs)),
        "observation_space": _space_size(getattr(base_env.cfg, "observation_space", 0)),
        "action_space": _space_size(getattr(base_env.cfg, "action_space", 0)),
        "action_scale": action_scale,
        "joint_names": joint_names,
        "action_semantics": getattr(
            base_env.cfg,
            "action_semantics",
            "joint_position_target_rad = action_normalized * action_scale",
        ),
        "action_delta_scale_rad": [float(value) * float(dt) for value in joint_velocity_safe],
        "action_clip": [-1.0, 1.0],
        "joint_velocity_safe_rad_s": joint_velocity_safe,
        "joint_acceleration_safe_rad_s2": joint_acceleration_safe,
        "joint_position_lower_rad": joint_position_lower,
        "joint_position_upper_rad": joint_position_upper,
        "legacy_policy_compatibility": "incompatible: retrain policies trained with absolute action targets",
        "rollout_schema_version": 2,
        "sim_reference": {
            "joint_position_field": "joint_position_after_rad",
            "joint_velocity_field": "joint_velocity_after_rad_s",
            "time_field": "sim_time_after_s",
            "source": "post-physics simulator state before done-environment auto-reset",
        },
    }

    if args_cli.export_policy:
        _export_policy(runner.agent, obs, export_dir, args_cli.export_onnx, metadata)
        if not args_cli.record_actions and args_cli.eval_episodes <= 0 and not args_cli.video:
            env.close()
            return

    if args_cli.record_actions:
        record_path = (
            os.path.abspath(args_cli.record_actions_path)
            if args_cli.record_actions_path
            else os.path.join(export_dir, f"rollouts_{args_cli.record_episodes}_episodes.json")
        )
        _record_action_rollouts(
            runner.agent,
            env,
            base_env,
            obs,
            record_path,
            args_cli.record_episodes,
            action_scale,
            float(dt),
            metadata,
            args_cli.record_max_steps,
        )
        env.close()
        return

    if args_cli.eval_episodes > 0:
        completed_episodes = 0
        successful_episodes = 0
        total_steps = 0
        start_time = time.time()
        print(f"[INFO] Evaluating {args_cli.eval_episodes} episodes headless...")
        while simulation_app.is_running() and completed_episodes < args_cli.eval_episodes:
            with torch.inference_mode():
                outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, terminated, truncated, extras = env.step(actions)

            done = (terminated | truncated).view(-1)
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                remaining = args_cli.eval_episodes - completed_episodes
                done_ids = done_ids[:remaining]
                success_buf = getattr(base_env, "_last_done_success", extras.get("success", None))
                if success_buf is None:
                    raise RuntimeError("Evaluation requires the environment to expose a success buffer.")
                successful_episodes += int(success_buf[done_ids].sum().item())
                completed_episodes += int(done_ids.numel())

                if completed_episodes % 100 == 0 or completed_episodes == args_cli.eval_episodes:
                    success_rate = 100.0 * successful_episodes / completed_episodes
                    print(
                        f"[EVAL] episodes={completed_episodes}/{args_cli.eval_episodes} "
                        f"successes={successful_episodes} success_rate={success_rate:.2f}%"
                    )
            total_steps += 1

        elapsed = time.time() - start_time
        success_rate = 100.0 * successful_episodes / max(completed_episodes, 1)
        print(
            f"[EVAL] done: episodes={completed_episodes} successes={successful_episodes} "
            f"success_rate={success_rate:.2f}% steps={total_steps} elapsed={elapsed:.2f}s"
        )
        env.close()
        return

    dashboard = None
    if not bool(getattr(args_cli, "headless", False)) and not args_cli.disable_play_ui and not args_cli.video:
        try:
            dashboard = _PlayDashboard(speed=args_cli.sim_speed)
            print("[INFO] Isaac Sim play dashboard enabled")
        except Exception as exc:
            print(f"[WARN] Isaac Sim play dashboard disabled: {exc}")

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        if dashboard is not None and not dashboard.should_step():
            try:
                simulation_app.update()
            except Exception:
                pass
            time.sleep(0.02)
            continue

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, rewards, terminated, truncated, extras = env.step(actions)
        timestep += 1

        if dashboard is not None:
            dashboard.update(
                _play_dashboard_state(
                    base_env,
                    actions,
                    rewards,
                    terminated,
                    truncated,
                    extras,
                    timestep,
                    float(dt),
                )
            )

        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation or interactive dashboard speed control.
        speed = dashboard.speed if dashboard is not None else max(0.0, float(args_cli.sim_speed))
        target_dt = dt / speed if speed > 0.0 else dt
        sleep_time = target_dt - (time.time() - start_time)
        if (args_cli.real_time or dashboard is not None) and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
