"""Isaac Lab environment bridge for TD-MPC2 data collection."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import torch

from . import helper
from .sequence_buffer import SequenceBuffer


@dataclass
class EpisodeTracker:
	obs: list[torch.Tensor] = field(default_factory=list)
	action: list[torch.Tensor] = field(default_factory=list)
	reward: list[torch.Tensor] = field(default_factory=list)
	terminated: list[torch.Tensor] = field(default_factory=list)
	constraint: list[torch.Tensor] = field(default_factory=list)

	def start(self, obs: torch.Tensor):
		self.reset()
		self.obs.append(obs.detach().cpu())

	def push(self, action, reward, terminated, constraint, next_obs):
		self.action.append(action.detach().cpu())
		self.reward.append(reward.detach().cpu().reshape(1))
		self.terminated.append(terminated.detach().cpu().reshape(1).float())
		self.constraint.append(constraint.detach().cpu().reshape(1))
		self.obs.append(next_obs.detach().cpu())

	def flush(self) -> tuple[torch.Tensor, ...] | None:
		if len(self.action) == 0:
			return None
		# Align to equal-length tensors for SequenceBuffer (drop final obs row).
		return (
			torch.stack(self.obs[:-1]),
			torch.stack(self.action),
			torch.stack(self.reward),
			torch.stack(self.terminated),
			torch.stack(self.constraint),
		)

	def reset(self):
		self.obs.clear()
		self.action.clear()
		self.reward.clear()
		self.terminated.clear()
		self.constraint.clear()


class TDMPC2VecEnv:
	"""Thin wrapper around Isaac Lab ManagerBasedRLEnv for TD-MPC2 rollout collection."""

	def __init__(self, env: gym.Env, device: str, action_scale: float = 0.30, obs_key: str = "policy"):
		self.env = env
		self.device = torch.device(device)
		self.action_scale = action_scale
		self.obs_key = obs_key
		self.num_envs = env.unwrapped.num_envs
		self._obs_buf: torch.Tensor | None = None
		self._action_dim: int | None = None
		self._obs_dim: int | None = None

	def _policy_obs(self, obs_dict) -> torch.Tensor:
		if isinstance(obs_dict, dict):
			obs = obs_dict[self.obs_key]
		else:
			obs = obs_dict
		return obs.to(self.device)

	@property
	def obs_dim(self) -> int:
		if self._obs_dim is None:
			raise RuntimeError("Call reset() before reading obs_dim.")
		return self._obs_dim

	@property
	def action_dim(self) -> int:
		if self._action_dim is None:
			raise RuntimeError("Call reset() before reading action_dim.")
		return self._action_dim

	def reset(self):
		obs_dict, _ = self.env.reset()
		self._obs_buf = self._policy_obs(obs_dict)
		self._obs_dim = self._obs_buf.shape[-1]
		self._action_dim = self.env.unwrapped.action_manager.total_action_dim
		return self._obs_buf

	def step(self, actions: torch.Tensor):
		"""Step with normalized TD-MPC2 actions in [-1, 1]."""
		actions = actions.to(self.device)
		obs_dict, reward, terminated, truncated, _ = self.env.step(actions * self.action_scale)
		obs = self._policy_obs(obs_dict)
		self._obs_buf = obs
		done = (terminated | truncated).to(self.device)
		return obs, reward.to(self.device), done, terminated.to(self.device)

	def random_action(self) -> torch.Tensor:
		return torch.empty(self.num_envs, self.action_dim, device=self.device).uniform_(-1, 1)


class RolloutCollector:
	"""Collect parallel-env episodes into a CPU ``SequenceBuffer``."""

	def __init__(self, env: TDMPC2VecEnv, buffer: SequenceBuffer):
		self.env = env
		self.buffer = buffer
		self.trackers = [EpisodeTracker() for _ in range(env.num_envs)]
		self.total_episodes = 0
		self.total_transitions = 0

	def reset_trackers(self, obs: torch.Tensor):
		for i, tr in enumerate(self.trackers):
			tr.start(obs[i])

	def ingest(
		self,
		prev_obs: torch.Tensor,
		actions: torch.Tensor,
		next_obs: torch.Tensor,
		rewards: torch.Tensor,
		dones: torch.Tensor,
	):
		constraint = helper.constraint_label_from_obs(next_obs)
		for i in range(self.env.num_envs):
			tr = self.trackers[i]
			if len(tr.obs) == 0:
				tr.start(prev_obs[i])
			tr.push(actions[i], rewards[i], dones[i], constraint[i], next_obs[i])
			if dones[i].item():
				ep = tr.flush()
				if ep is not None:
					self.buffer.add_episode(*ep)
					self.total_episodes += 1
					self.total_transitions += ep[0].shape[0]
				tr.start(next_obs[i])
		self.total_transitions += self.env.num_envs

	def ready_for_update(self, min_size: int) -> bool:
		return len(self.buffer) >= min_size
