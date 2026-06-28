from __future__ import annotations

import argparse


def add_tdmpc2_args(parser: argparse.ArgumentParser):
	arg_group = parser.add_argument_group("tdmpc2", description="Arguments for TD-MPC2 world-model training.")
	arg_group.add_argument("--experiment_name", type=str, default=None)
	arg_group.add_argument("--run_name", type=str, default=None)
	arg_group.add_argument("--max_iterations", type=int, default=None)
	arg_group.add_argument("--seed_steps", type=int, default=None)
	arg_group.add_argument("--resume", action="store_true", default=False)
	arg_group.add_argument("--load_run", type=str, default=None)
	arg_group.add_argument("--checkpoint", type=str, default=None)
	arg_group.add_argument("--mpc", action="store_true", default=False, help="Use MPPI during collection (slow).")
	arg_group.add_argument("--wm_num_envs", type=int, default=None, help="Override num_envs for WM training (VRAM).")


def update_tdmpc2_cfg(agent_cfg, args_cli: argparse.Namespace):
	if args_cli.seed is not None:
		agent_cfg.seed = args_cli.seed
	if args_cli.experiment_name is not None:
		agent_cfg.experiment_name = args_cli.experiment_name
	if args_cli.run_name is not None:
		agent_cfg.run_name = args_cli.run_name
	if args_cli.max_iterations is not None:
		agent_cfg.max_iterations = args_cli.max_iterations
	if args_cli.seed_steps is not None:
		agent_cfg.seed_steps = args_cli.seed_steps
	if args_cli.resume:
		agent_cfg.resume = True
	if args_cli.load_run is not None:
		agent_cfg.load_run = args_cli.load_run
	if args_cli.checkpoint is not None:
		agent_cfg.load_checkpoint = args_cli.checkpoint
	if args_cli.mpc:
		agent_cfg.mpc = True
	return agent_cfg
