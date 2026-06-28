"""CPU-backed sequence replay buffer for Kuavo 115-dim state trajectories."""

from __future__ import annotations

from typing import Optional

import torch

from . import helper


class SequenceBuffer:
	"""
	Host-memory replay buffer for 115-dim observation sequences.

	Storage lives entirely on CPU; ``sample()`` asynchronously transfers mini-batches
	to the target GPU device only at training time (8GB VRAM protection).
	"""

	def __init__(
		self,
		capacity: int = 1_000_000,
		sequence_length: int = 50,
		batch_size: int = 256,
		obs_dim: int | None = None,
		action_dim: int = helper.ACTION_DIM,
		obs_profile: str = "dance",
		device: str | torch.device = "cuda:0",
	):
		self.capacity = capacity
		self.sequence_length = sequence_length
		self.batch_size = batch_size
		self.obs_dim = obs_dim or helper.get_obs_profile(obs_profile).obs_dim
		self.action_dim = action_dim
		self.obs_profile = obs_profile
		self.storage_device = torch.device("cpu")
		self.sample_device = torch.device(device)

		self._obs = torch.zeros(capacity, sequence_length, self.obs_dim, device=self.storage_device)
		self._action = torch.zeros(capacity, sequence_length, self.action_dim, device=self.storage_device)
		self._reward = torch.zeros(capacity, sequence_length, 1, device=self.storage_device)
		self._terminated = torch.zeros(capacity, sequence_length, 1, device=self.storage_device)
		self._constraint = torch.zeros(capacity, sequence_length, 1, device=self.storage_device)
		self._mask = torch.zeros(capacity, sequence_length, 1, device=self.storage_device)

		self._default_obs = helper.default_stance_obs(profile=obs_profile, device=self.storage_device)
		self._write_ptr = 0
		self._size = 0
		self._episode_boundaries: list[tuple[int, int]] = []

	def __len__(self) -> int:
		return self._size

	@property
	def num_episodes(self) -> int:
		return len(self._episode_boundaries)

	def _pad_episode(
		self,
		obs: torch.Tensor,
		action: torch.Tensor,
		reward: torch.Tensor,
		terminated: torch.Tensor,
		constraint: Optional[torch.Tensor] = None,
	) -> tuple[torch.Tensor, ...]:
		"""Pad a short episode to ``sequence_length`` with default stance obs and validity mask."""
		t = obs.shape[0]
		seq = self.sequence_length

		if constraint is None:
			constraint = helper.constraint_label_from_obs(obs)

		if t >= seq:
			s = obs[-seq:]
			a = action[-seq:]
			r = reward[-seq:]
			term = terminated[-seq:]
			cons = constraint[-seq:]
			mask = torch.ones(seq, 1, device=self.storage_device)
			return s, a, r, term, cons, mask

		pad_len = seq - t
		pad_obs = self._default_obs.unsqueeze(0).expand(pad_len, -1)
		pad_action = torch.zeros(pad_len, self.action_dim, device=self.storage_device)
		pad_reward = torch.zeros(pad_len, 1, device=self.storage_device)
		pad_terminated = torch.ones(pad_len, 1, device=self.storage_device)
		pad_constraint = torch.ones(pad_len, 1, device=self.storage_device)
		valid_mask = torch.cat([
			torch.ones(t, 1, device=self.storage_device),
			torch.zeros(pad_len, 1, device=self.storage_device),
		])

		return (
			torch.cat([obs.to(self.storage_device), pad_obs]),
			torch.cat([action.to(self.storage_device), pad_action]),
			torch.cat([reward.to(self.storage_device).reshape(-1, 1), pad_reward]),
			torch.cat([terminated.to(self.storage_device).reshape(-1, 1), pad_terminated]),
			torch.cat([constraint.to(self.storage_device).reshape(-1, 1), pad_constraint]),
			valid_mask,
		)

	def add_episode(
		self,
		obs: torch.Tensor,
		action: torch.Tensor,
		reward: torch.Tensor,
		terminated: torch.Tensor,
		constraint: Optional[torch.Tensor] = None,
	) -> int:
		"""Add one episode (variable length) with zero-padding if shorter than sequence_length."""
		obs, action, reward, terminated, constraint, mask = self._pad_episode(
			obs, action, reward, terminated, constraint
		)
		idx = self._write_ptr
		self._obs[idx] = obs
		self._action[idx] = action
		self._reward[idx] = reward
		self._terminated[idx] = terminated
		self._constraint[idx] = constraint
		self._mask[idx] = mask

		self._episode_boundaries.append((idx, obs.shape[0]))
		self._write_ptr = (self._write_ptr + 1) % self.capacity
		self._size = min(self._size + 1, self.capacity)
		return idx

	def add(self, td: dict) -> int:
		"""Convenience wrapper accepting a dict with obs/action/reward/terminated keys."""
		return self.add_episode(
			obs=td["obs"],
			action=td["action"],
			reward=td["reward"],
			terminated=td.get("terminated", torch.zeros(td["obs"].shape[0])),
			constraint=td.get("constraint"),
		)

	def sample(self, batch_size: Optional[int] = None):
		"""
		Randomly sample ``[Batch, Sequence_Length, 115]`` observation sequences.

		Returns tensors on ``sample_device`` (GPU) via non-blocking transfer.
		"""
		if self._size == 0:
			raise RuntimeError("SequenceBuffer is empty; add episodes before sampling.")

		bs = batch_size or self.batch_size
		indices = torch.randint(0, self._size, (bs,), device=self.storage_device)

		obs = self._obs[indices].to(self.sample_device, non_blocking=True)
		action = self._action[indices].to(self.sample_device, non_blocking=True)
		reward = self._reward[indices].to(self.sample_device, non_blocking=True)
		terminated = self._terminated[indices].to(self.sample_device, non_blocking=True)
		constraint = self._constraint[indices].to(self.sample_device, non_blocking=True)
		mask = self._mask[indices].to(self.sample_device, non_blocking=True)

		return obs, action, reward, terminated, constraint, mask

	def sample_horizon(self, horizon: int, batch_size: Optional[int] = None):
		"""
		Sample contiguous sub-sequences of length ``horizon+1`` for TD-MPC2 latent rollout.

		Returns obs shaped ``[horizon+1, batch, 115]`` on GPU.
		"""
		if self._size == 0:
			raise RuntimeError("SequenceBuffer is empty; add episodes before sampling.")

		bs = batch_size or self.batch_size
		h1 = horizon + 1
		obs_batch = []
		action_batch = []
		reward_batch = []
		term_batch = []
		constraint_batch = []
		mask_batch = []

		for _ in range(bs):
			ep_idx = torch.randint(0, self._size, (1,)).item()
			seq_obs = self._obs[ep_idx]
			valid_len = int(self._mask[ep_idx].sum().item())
			max_start = max(valid_len - h1, 0)
			start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0

			end = start + h1
			obs_slice = seq_obs[start:end]
			act_slice = self._action[ep_idx, start : start + horizon]
			rew_slice = self._reward[ep_idx, start + 1 : end]
			term_slice = self._terminated[ep_idx, start + 1 : end]
			cons_slice = self._constraint[ep_idx, start + 1 : end]
			mask_slice = self._mask[ep_idx, start + 1 : end]

			if obs_slice.shape[0] < h1:
				pad = h1 - obs_slice.shape[0]
				obs_slice = torch.cat([obs_slice, self._default_obs.unsqueeze(0).expand(pad, -1)])
				act_pad = torch.zeros(pad, self.action_dim, device=self.storage_device)
				act_slice = torch.cat([act_slice, act_pad])
				rew_pad = torch.zeros(pad, 1, device=self.storage_device)
				rew_slice = torch.cat([rew_slice, rew_pad])
				term_pad = torch.ones(pad, 1, device=self.storage_device)
				term_slice = torch.cat([term_slice, term_pad])
				cons_pad = torch.ones(pad, 1, device=self.storage_device)
				cons_slice = torch.cat([cons_slice, cons_pad])
				mask_pad = torch.zeros(pad, 1, device=self.storage_device)
				mask_slice = torch.cat([mask_slice, mask_pad])

			obs_batch.append(obs_slice)
			action_batch.append(act_slice[:horizon])
			reward_batch.append(rew_slice[:horizon])
			term_batch.append(term_slice[:horizon])
			constraint_batch.append(cons_slice[:horizon])
			mask_batch.append(mask_slice[:horizon])

		obs = torch.stack(obs_batch).to(self.sample_device, non_blocking=True).permute(1, 0, 2)
		action = torch.stack(action_batch).to(self.sample_device, non_blocking=True).permute(1, 0, 2)
		reward = torch.stack(reward_batch).to(self.sample_device, non_blocking=True).permute(1, 0, 2)
		terminated = torch.stack(term_batch).to(self.sample_device, non_blocking=True).permute(1, 0, 2)
		constraint = torch.stack(constraint_batch).to(self.sample_device, non_blocking=True).permute(1, 0, 2)
		mask = torch.stack(mask_batch).to(self.sample_device, non_blocking=True).permute(1, 0, 2)

		return obs, action, reward, terminated, constraint, mask
