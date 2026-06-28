"""TD-MPC2 training runner: collect rollouts in Isaac Lab, update world model on CPU buffer."""

from __future__ import annotations

import os
import time
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

from . import helper
from .env_bridge import RolloutCollector, TDMPC2VecEnv
from .sequence_buffer import SequenceBuffer
from .tdmpc2 import TDMPC2


class TDMPC2Runner:
	def __init__(
		self,
		env: TDMPC2VecEnv,
		agent_cfg,
		log_dir: str,
		device: str = "cuda:0",
	):
		self.env = env
		self.agent_cfg = agent_cfg
		self.log_dir = log_dir
		self.device = device

		_ = env.reset()
		obs_profile = getattr(agent_cfg, "obs_profile", "dance")
		prof = helper.get_obs_profile(obs_profile)
		if env.obs_dim != prof.obs_dim:
			print(f"[TDMPC2] Warning: env obs_dim={env.obs_dim} != profile {prof.name}={prof.obs_dim}; using env dim.")

		self.tdmpc_cfg = helper.default_cfg(
			obs_profile=obs_profile,
			action_scale=getattr(agent_cfg, "action_scale", 0.30),
			batch_size=getattr(agent_cfg, "batch_size", 256),
			horizon=getattr(agent_cfg, "horizon", 5),
			sequence_length=getattr(agent_cfg, "sequence_length", 50),
			buffer_size=getattr(agent_cfg, "buffer_size", 500_000),
			mpc=getattr(agent_cfg, "mpc", False),
			num_samples=getattr(agent_cfg, "num_samples", 256),
			num_elites=getattr(agent_cfg, "num_elites", 32),
			num_pi_trajs=getattr(agent_cfg, "num_pi_trajs", 12),
			latent_dim=getattr(agent_cfg, "latent_dim", 512),
			lr=getattr(agent_cfg, "learning_rate", 3e-4),
		)
		self.tdmpc_cfg.obs_shape = (env.obs_dim,)
		self.tdmpc_cfg.action_dim = env.action_dim
		self.tdmpc_cfg.action_shape = (env.action_dim,)

		self.agent = TDMPC2(self.tdmpc_cfg)
		self.buffer = SequenceBuffer(
			capacity=self.tdmpc_cfg.buffer_size,
			sequence_length=self.tdmpc_cfg.sequence_length,
			batch_size=self.tdmpc_cfg.batch_size,
			obs_dim=env.obs_dim,
			action_dim=env.action_dim,
			obs_profile=obs_profile,
			device=device,
		)
		self.collector = RolloutCollector(env, self.buffer)
		self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"), flush_secs=10)
		self.global_step = 0

	def save(self, path: str):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		self.agent.save(path)

	def load(self, path: str):
		self.agent.load(path)

	def learn(self, num_iterations: int, init_at_random_ep_len: bool = True):
		seed_steps = getattr(self.agent_cfg, "seed_steps", 5000)
		steps_per_iter = getattr(self.agent_cfg, "steps_per_env", 24)
		updates_per_iter = getattr(self.agent_cfg, "updates_per_iter", 1)
		min_buffer = getattr(self.agent_cfg, "min_buffer_size", self.tdmpc_cfg.batch_size * 2)
		log_interval = getattr(self.agent_cfg, "log_interval", 10)
		save_interval = getattr(self.agent_cfg, "save_interval", 100)

		obs = self.env.reset()
		self.collector.reset_trackers(obs)
		if init_at_random_ep_len and hasattr(self.env.env.unwrapped, "episode_length_buf"):
			max_len = self.env.env.unwrapped.max_episode_length
			self.env.env.unwrapped.episode_length_buf[:] = torch.randint(
				0, max_len, (self.env.num_envs,), device=self.env.env.unwrapped.device
			)

		print(f"[TDMPC2] obs_dim={self.env.obs_dim}, action_dim={self.env.action_dim}, profile={self.tdmpc_cfg.obs_profile}")
		print(f"[TDMPC2] seed_steps={seed_steps}, steps_per_iter={steps_per_iter}, mpc={self.tdmpc_cfg.mpc}")

		for it in range(num_iterations):
			t0 = time.time()
			for _ in range(steps_per_iter):
				if self.global_step < seed_steps:
					actions = self.env.random_action()
				else:
					actions = self.agent.act_batch(obs, use_mpc=self.tdmpc_cfg.mpc and self.env.num_envs == 1)

				next_obs, rewards, dones, terminated = self.env.step(actions)
				self.collector.ingest(obs, actions, next_obs, rewards, dones)
				obs = next_obs
				self.global_step += self.env.num_envs

			update_info = None
			if self.collector.ready_for_update(min_buffer):
				for _ in range(updates_per_iter):
					update_info = self.agent.update(self.buffer)

			if update_info is not None and (it + 1) % log_interval == 0:
				for key, val in update_info.items():
					if hasattr(val, "item"):
						self.writer.add_scalar(f"train/{key}", val.item(), it)
				self.writer.add_scalar("train/buffer_size", len(self.buffer), it)
				self.writer.add_scalar("train/episodes_collected", self.collector.total_episodes, it)
				self.writer.add_scalar("train/global_steps", self.global_step, it)
				self.writer.add_scalar("train/fps", self.env.num_envs * steps_per_iter / max(time.time() - t0, 1e-6), it)

			if (it + 1) % save_interval == 0:
				self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))

			if (it + 1) % log_interval == 0:
				msg = f"[TDMPC2] iter {it + 1}/{num_iterations} | buffer={len(self.buffer)} | steps={self.global_step}"
				if update_info is not None:
					msg += f" | loss={update_info.get('total_loss', 0):.4f}"
				print(msg)

		self.save(os.path.join(self.log_dir, "model_final.pt"))
		self.writer.close()

	def evaluate(self, num_steps: int = 500, use_mpc: bool = True):
		"""Run closed-loop evaluation with optional MPPI (single-env recommended)."""
		obs = self.env.reset()
		rewards = []
		for _ in range(num_steps):
			actions = self.agent.act_batch(obs, eval_mode=True, use_mpc=use_mpc)
			obs, reward, dones, _ = self.env.step(actions)
			rewards.append(reward.mean().item())
			if dones.any():
				break
		return {"mean_reward": sum(rewards) / max(len(rewards), 1), "steps": len(rewards)}
