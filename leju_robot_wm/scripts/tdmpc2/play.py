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
parser.add_argument("--eval_mode", action="store_true", help="Use policy mean (deterministic). Default: stochastic, same as training.")
parser.add_argument("--zero_action", action="store_true", help="Ignore checkpoint; hold default pose (action=0 baseline).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", default=False, help="Record video (headless-friendly).")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in env steps.")
cli_args.add_tdmpc2_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if not args_cli.checkpoint and not args_cli.zero_action:
	parser.error("--checkpoint is required for play (or pass --zero_action for baseline).")

if args_cli.zero_action:
	args_cli.checkpoint = args_cli.checkpoint or "zero_action_baseline"

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


def _format_termination_reason(env) -> str:
	"""Return which termination term(s) fired (e.g. base_contact / dof_pos_illegal)."""
	try:
		base = env.unwrapped
		tm = base.termination_manager
		parts = []
		for name in tm.active_terms:
			val = tm.get_term(name)
			if torch.is_tensor(val) and bool(val.any().item()):
				parts.append(name)
			elif not torch.is_tensor(val) and bool(val):
				parts.append(name)
		# also show raw last terminated flags if available
		if hasattr(tm, "get_active_iterable_terms"):
			pass
		return ",".join(parts) if parts else "unknown(done)"
	except Exception as exc:  # noqa: BLE001 — diagnostic only
		return f"unavailable({exc})"


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

	checkpoint = os.path.abspath(args_cli.checkpoint) if not args_cli.zero_action else None
	log_dir = os.path.dirname(checkpoint) if checkpoint else os.path.join("logs", "tdmpc2", "baseline")

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

	wm_env = TDMPC2VecEnv(env, device=agent_cfg.device, action_scale=getattr(agent_cfg, "action_scale", 1.0))
	runner = TDMPC2Runner(wm_env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)
	if not args_cli.zero_action:
		runner.load(checkpoint)

	use_mpc = agent_cfg.mpc
	max_steps = args_cli.video_length if args_cli.video else args_cli.eval_steps
	# Video needs full length; stop-on-done truncates RecordVideo to a few frames (~1s).
	stop_on_done = not args_cli.video
	mode = "zero_action" if args_cli.zero_action else f"mpc={use_mpc}, eval_mode={args_cli.eval_mode}"
	print(f"[INFO] Evaluating with {mode}, steps={max_steps}, video={args_cli.video}, stop_on_done={stop_on_done}")

	obs = wm_env.reset()
	rewards = []
	timestep = 0
	first_done_step = None
	num_resets = 0
	while simulation_app.is_running() and timestep < max_steps:
		with torch.inference_mode():
			if args_cli.zero_action:
				actions = wm_env.zero_action()
			else:
				actions = runner.agent.act_batch(obs, eval_mode=args_cli.eval_mode, use_mpc=use_mpc)
			obs, reward, dones, _ = wm_env.step(actions)
		rewards.append(reward.mean().item())
		timestep += 1
		if dones.any():
			if first_done_step is None:
				first_done_step = timestep
				sim_s = first_done_step * float(env_cfg.sim.dt) * getattr(env_cfg, "decimation", 4)
				term_reason = _format_termination_reason(env)
				print(
					f"[INFO] Episode ended at step {first_done_step} "
					f"(~{sim_s:.2f}s sim). reason={term_reason}. "
					+ ("Continuing until video_length…" if not stop_on_done else "Stopping (no video).")
				)
			num_resets += 1
			if stop_on_done:
				break

	metrics = {
		"mean_reward": sum(rewards) / max(len(rewards), 1),
		"steps": len(rewards),
		"first_done_step": first_done_step,
		"num_resets": num_resets,
	}
	print(f"[INFO] Eval metrics: {metrics}")
	if first_done_step is not None and first_done_step < max_steps:
		print(
			"[INFO] Early stop is a termination (not timeout). "
			"Look at reason= above: dof_pos_illegal = ankle/actuator NaN (no visual fall); "
			"base_contact = torso hit something; time_out = survived."
		)
	if args_cli.video:
		print(f"[INFO] Video saved under: {os.path.join(log_dir, 'videos', 'play')}")
	env.close()


if __name__ == "__main__":
	main()
	simulation_app.close()
