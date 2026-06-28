# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate / play TD-MPC2 with MPPI in Isaac Lab."""

import argparse
import os
import sys

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play TD-MPC2 policy with MPPI in simulation.")
parser.add_argument("--task", type=str, required=True, help="TDMPC2-Play gym task id.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--no_mpc", action="store_true", help="Use policy prior only (no MPPI).")
parser.add_argument("--mpc", action="store_true", help="Enable MPPI planning (overrides cfg default).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", default=False, help="Record video (headless-friendly).")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in env steps.")
cli_args.add_tdmpc2_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if not args_cli.checkpoint:
	parser.error("--checkpoint is required for play.")

if args_cli.video:
	args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

import ext_template.tasks  # noqa: F401
from ext_template.tasks.locomotion.velocity.world_model_core.env_bridge import TDMPC2VecEnv
from ext_template.tasks.locomotion.velocity.world_model_core.runner import TDMPC2Runner


@hydra_task_config(args_cli.task, "tdmpc2_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
	agent_cfg = cli_args.update_tdmpc2_cfg(agent_cfg, args_cli)
	if args_cli.no_mpc:
		agent_cfg.mpc = False
	elif args_cli.mpc:
		agent_cfg.mpc = True

	if args_cli.num_envs is not None:
		env_cfg.scene.num_envs = args_cli.num_envs
	else:
		# Play / MPPI 均按单 env 评测（cfg PLAY 默认亦 num_envs=1）
		env_cfg.scene.num_envs = 1
	env_cfg.seed = args_cli.seed
	env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

	checkpoint = os.path.abspath(args_cli.checkpoint)
	log_dir = os.path.dirname(checkpoint)

	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
	if args_cli.video:
		video_dir = os.path.join(log_dir, "videos", "play")
		video_kwargs = {
			"video_folder": video_dir,
			"step_trigger": lambda step: step == 0,
			"video_length": args_cli.video_length,
			"disable_logger": True,
		}
		print("[INFO] Recording video during play.")
		print_dict(video_kwargs, nesting=4)
		env = gym.wrappers.RecordVideo(env, **video_kwargs)

	if isinstance(env.unwrapped, DirectMARLEnv):
		env = multi_agent_to_single_agent(env)

	wm_env = TDMPC2VecEnv(env, device=agent_cfg.device, action_scale=getattr(agent_cfg, "action_scale", 0.30))
	runner = TDMPC2Runner(wm_env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)
	runner.load(checkpoint)

	use_mpc = agent_cfg.mpc
	max_steps = args_cli.video_length if args_cli.video else args_cli.eval_steps
	print(f"[INFO] Evaluating with mpc={use_mpc}, steps={max_steps}, video={args_cli.video}")

	obs = wm_env.reset()
	rewards = []
	timestep = 0
	while simulation_app.is_running() and timestep < max_steps:
		with torch.inference_mode():
			actions = runner.agent.act_batch(obs, eval_mode=True, use_mpc=use_mpc)
			obs, reward, dones, _ = wm_env.step(actions)
		rewards.append(reward.mean().item())
		timestep += 1
		if dones.any():
			break

	metrics = {"mean_reward": sum(rewards) / max(len(rewards), 1), "steps": len(rewards)}
	print(f"[INFO] Eval metrics: {metrics}")
	if args_cli.video:
		print(f"[INFO] Video saved under: {os.path.join(log_dir, 'videos', 'play')}")
	env.close()


if __name__ == "__main__":
	main()
	simulation_app.close()
