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


def _success_buffer(base_env, extras):
    success = getattr(base_env, "_last_done_success", None)
    if success is None and isinstance(extras, dict):
        success = extras.get("success", None)
    return success


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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
            joint_targets = actions * action_scale

            step_records = []
            for env_id in range(num_envs):
                record = {
                    "step": episode_step[env_id],
                    "time_s": episode_step[env_id] * dt,
                    "observation": _tensor_row_to_list(obs, env_id),
                    "action_normalized": _tensor_row_to_list(actions, env_id),
                    "joint_position_target_rad": _tensor_row_to_list(joint_targets, env_id),
                    "joint_position_before_rad": _tensor_row_to_list(joint_pos, env_id),
                    "joint_velocity_before_rad_s": _tensor_row_to_list(joint_vel, env_id),
                }
                buffers[env_id].append(record)
                step_records.append(record)

            obs, rewards, terminated, truncated, extras = env.step(actions)

        done = (terminated | truncated).view(-1)
        success = _success_buffer(base_env, extras)
        for env_id, record in enumerate(step_records):
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
    metadata = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "log_dir": log_dir,
        "ml_framework": args_cli.ml_framework,
        "algorithm": algorithm,
        "dt_s": float(dt),
        "num_envs": int(getattr(env, "num_envs", base_env.num_envs)),
        "observation_space": int(getattr(base_env.cfg, "observation_space", 0)),
        "action_space": int(getattr(base_env.cfg, "action_space", 0)),
        "action_scale": action_scale,
        "joint_names": joint_names,
        "action_semantics": "joint_position_target_rad = action_normalized * action_scale",
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

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

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
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
