# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train TD-MPC2 world model with Isaac Lab Kuavo environments."""

import argparse
import os
import sys
from datetime import datetime

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train TD-MPC2 world model on Kuavo Isaac Lab tasks.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Gym task id (must include TDMPC2 in name).")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
cli_args.add_tdmpc2_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

import ext_template.tasks  # noqa: F401
from ext_template.tasks.locomotion.velocity.world_model_core.env_bridge import TDMPC2VecEnv
from ext_template.tasks.locomotion.velocity.world_model_core.runner import TDMPC2Runner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "tdmpc2_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
	agent_cfg = cli_args.update_tdmpc2_cfg(agent_cfg, args_cli)

	if args_cli.num_envs is not None:
		env_cfg.scene.num_envs = args_cli.num_envs
	elif args_cli.wm_num_envs is not None:
		env_cfg.scene.num_envs = args_cli.wm_num_envs
	else:
		# 8GB laptop safe default for world-model + PhysX co-training
		env_cfg.scene.num_envs = min(getattr(env_cfg.scene, "num_envs", 512), 512)

	env_cfg.seed = agent_cfg.seed
	env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

	log_root = os.path.abspath(os.path.join("logs", "tdmpc2", agent_cfg.experiment_name))
	log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	if agent_cfg.run_name:
		log_dir += f"_{agent_cfg.run_name}"
	log_dir = os.path.join(log_root, log_dir)
	os.makedirs(log_dir, exist_ok=True)
	print(f"[INFO] TD-MPC2 log dir: {log_dir}")

	env = gym.make(args_cli.task, cfg=env_cfg)
	if isinstance(env.unwrapped, DirectMARLEnv):
		env = multi_agent_to_single_agent(env)

	device = agent_cfg.device
	wm_env = TDMPC2VecEnv(env, device=device, action_scale=getattr(agent_cfg, "action_scale", 0.30))
	runner = TDMPC2Runner(wm_env, agent_cfg, log_dir=log_dir, device=device)

	dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
	dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
	dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
	dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

	if getattr(agent_cfg, "resume", False) and agent_cfg.load_checkpoint:
		ckpt = agent_cfg.load_checkpoint
		if not os.path.isabs(ckpt):
			ckpt = os.path.join(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
		print(f"[INFO] Loading checkpoint: {ckpt}")
		runner.load(ckpt)

	runner.learn(num_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
	env.close()


if __name__ == "__main__":
	main()
	simulation_app.close()
