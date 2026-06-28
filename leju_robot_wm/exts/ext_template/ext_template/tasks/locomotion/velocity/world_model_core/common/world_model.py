"""Kuavo S49 TD-MPC2 implicit world model (state-only, with constraint decoder)."""

from copy import deepcopy

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

from .. import helper
from . import layers


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model for Kuavo 115-dim state / 26-dim action.
	Includes a Constraint/Termination decoder to penalize fallen (躺平) behaviours.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		assert cfg.action_dim == cfg.action_shape[0]

		self._encoder = layers.state_encoder(cfg)
		self._dynamics = layers.mlp(
			cfg.latent_dim + cfg.action_dim, 2 * [cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg)
		)
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 1)
		self._constraint = layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 1)
		self._pi = layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
		self._Qs = layers.Ensemble([
			layers.mlp(
				cfg.latent_dim + cfg.action_dim,
				2 * [cfg.mlp_dim],
				max(cfg.num_bins, 1),
				dropout=cfg.dropout,
			)
			for _ in range(cfg.num_q)
		])
		self.apply(helper.weight_init)
		helper.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = "Kuavo TD-MPC2 World Model\n"
		modules = ["State Encoder", "Dynamics", "Reward", "Termination", "Constraint", "Policy prior", "Q-functions"]
		for i, m in enumerate([
			self._encoder, self._dynamics, self._reward, self._termination,
			self._constraint, self._pi, self._Qs,
		]):
			repr += f"{modules[i]}: {m}\n"
		repr += f"Learnable parameters: {self.total_params:,}"
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def encode(self, obs):
		"""Encode 115-dim state observation(s) into latent representation."""
		return self._encoder(obs)

	def next(self, z, a):
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a):
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def termination(self, z, unnormalized=False):
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))

	def constraint(self, z, unnormalized=False):
		"""Predict base_contact OR upright_factor violation from latent state."""
		if unnormalized:
			return self._constraint(z)
		return torch.sigmoid(self._constraint(z))

	def pi(self, z):
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = helper.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)
		log_prob = helper.gaussian_logprob(eps, log_std)
		scaled_log_prob = log_prob * eps.shape[-1]
		action = mean + eps * log_std.exp()
		mean, action, log_prob = helper.squash(mean, action, log_prob)
		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.0,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, return_type="min", target=False, detach=False):
		assert return_type in {"min", "avg", "all"}
		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)
		if return_type == "all":
			return out
		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = helper.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
