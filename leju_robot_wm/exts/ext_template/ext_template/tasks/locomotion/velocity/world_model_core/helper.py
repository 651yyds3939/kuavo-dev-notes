"""Kuavo TD-MPC2 helpers: obs profiles, math, init, and constraint utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

ACTION_DIM = 26
ACTION_SHAPE = (ACTION_DIM,)

UPRIGHT_MIN = 0.75
CONSTRAINT_PENALTY = 50.0
SLICE_GRAVITY = slice(3, 6)


@dataclass(frozen=True)
class ObsProfile:
	"""Observation layout for Kuavo policy groups (matches deploy / punch_env_cfg order)."""

	name: str
	obs_dim: int
	slice_gravity: slice = SLICE_GRAVITY
	has_reference: bool = False
	has_phase: bool = False

	def default_stance_obs(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
		obs = torch.zeros(self.obs_dim, device=device, dtype=dtype)
		obs[self.slice_gravity] = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
		if self.has_phase:
			obs[-2:] = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
		return obs


OBS_PROFILES: dict[str, ObsProfile] = {
	"dance": ObsProfile(
		name="dance",
		obs_dim=115,
		has_reference=True,
		has_phase=True,
	),
	"velocity": ObsProfile(
		name="velocity",
		obs_dim=87,
		has_reference=False,
		has_phase=False,
	),
}

# Backward-compatible aliases (dance / S49)
OBS_DIM = OBS_PROFILES["dance"].obs_dim
OBS_SHAPE = (OBS_DIM,)


def get_obs_profile(name: str) -> ObsProfile:
	if name not in OBS_PROFILES:
		raise KeyError(f"Unknown obs profile '{name}'. Choose from {list(OBS_PROFILES)}")
	return OBS_PROFILES[name]


def default_stance_obs(
	obs_dim: int | None = None,
	profile: str = "dance",
	device: torch.device | str = "cpu",
	dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	prof = get_obs_profile(profile) if obs_dim is None else None
	if obs_dim is not None and prof is not None and prof.obs_dim != obs_dim:
		prof = ObsProfile(name="custom", obs_dim=obs_dim, has_phase=(obs_dim >= 115))
	elif prof is None:
		prof = ObsProfile(name="custom", obs_dim=obs_dim or OBS_DIM, has_phase=(obs_dim or OBS_DIM) >= 115)
	return prof.default_stance_obs(device, dtype)


def upright_factor_from_obs(obs: torch.Tensor, min_upright: float = UPRIGHT_MIN) -> torch.Tensor:
	z = -obs[..., SLICE_GRAVITY][..., 2]
	return torch.clamp((z - min_upright) / max(1.0 - min_upright, 1e-6), 0.0, 1.0)


def constraint_label_from_obs(
	obs: torch.Tensor,
	base_contact: torch.Tensor | None = None,
) -> torch.Tensor:
	fallen = upright_factor_from_obs(obs) < UPRIGHT_MIN
	if base_contact is not None:
		violated = fallen | base_contact.squeeze(-1).bool()
	else:
		violated = fallen
	return violated.float().unsqueeze(-1)


def soft_ce(pred, target, cfg):
	pred = F.log_softmax(pred, dim=-1)
	target = two_hot(target, cfg)
	return -(target * pred).sum(-1, keepdim=True)


def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
	residual = -0.5 * eps.pow(2) - log_std
	log_prob = residual - 0.9189385175704956
	return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
	log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
	return mu, pi, log_pi


def symlog(x):
	return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
	if cfg.num_bins == 0:
		return x
	if cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(-1)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
	bin_idx = bin_idx.long()
	soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
	return soft_two_hot


def two_hot_inv(x, cfg):
	if cfg.num_bins == 0:
		return x
	if cfg.num_bins == 1:
		return symexp(x)
	dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
	return symexp(x)


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
	logits = p.log()
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)
	gumbels = (logits + gumbels) / temperature
	y_soft = gumbels.softmax(dim)
	return y_soft.argmax(-1)


def termination_statistics(pred, target, eps=1e-9):
	pred = pred.squeeze(-1)
	target = target.squeeze(-1)
	rate = target.sum() / len(target)
	tp = ((pred > 0.5) & (target == 1)).sum()
	fn = ((pred <= 0.5) & (target == 1)).sum()
	fp = ((pred > 0.5) & (target == 0)).sum()
	recall = tp / (tp + fn + eps)
	precision = tp / (tp + fp + eps)
	f1 = 2 * (precision * recall) / (precision + recall + eps)
	return TensorDict({"termination_rate": rate, "termination_f1": f1})


def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)


def zero_(params):
	for p in params:
		p.data.fill_(0)


class RunningScale(torch.nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.register_buffer("value", torch.ones(1, dtype=torch.float32))
		self.register_buffer("_percentiles", torch.tensor([5, 95], dtype=torch.float32))

	def _positions(self, x_shape, device=None):
		pct = self._percentiles if device is None else self._percentiles.to(device)
		positions = pct * (x_shape - 1) / 100
		floored = torch.floor(positions)
		ceiled = floored + 1
		ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
		weight_ceiled = positions - floored
		weight_floored = 1.0 - weight_ceiled
		return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

	def _percentile(self, x):
		x_dtype, x_shape = x.dtype, x.shape
		x = x.flatten(1, x.ndim - 1)
		in_sorted = torch.sort(x, dim=0).values
		floored, ceiled, weight_floored, weight_ceiled = self._positions(x.shape[0], x.device)
		d0 = in_sorted[floored] * weight_floored
		d1 = in_sorted[ceiled] * weight_ceiled
		return (d0 + d1).reshape(-1, *x_shape[1:]).to(x_dtype)

	def update(self, x):
		percentiles = self._percentile(x.detach())
		value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
		self.value.data.lerp_(value, self.cfg.tau)

	def forward(self, x, update=False):
		if update:
			self.update(x)
		return x / self.value


@dataclass
class KuavoTDMPC2Config:
	obs_profile: str = "dance"
	obs_shape: tuple = field(default_factory=lambda: (OBS_PROFILES["dance"].obs_dim,))
	action_shape: tuple = ACTION_SHAPE
	action_dim: int = ACTION_DIM
	action_scale: float = 0.30
	latent_dim: int = 512
	enc_dim: int = 256
	mlp_dim: int = 512
	num_enc_layers: int = 2
	num_bins: int = 101
	vmin: float = -10.0
	vmax: float = 10.0
	bin_size: float = field(init=False)
	simnorm_dim: int = 8
	dropout: float = 0.01
	num_q: int = 5
	lr: float = 3e-4
	enc_lr_scale: float = 0.1
	tau: float = 0.01
	grad_clip_norm: float = 10.0
	rho: float = 0.5
	consistency_coef: float = 2.0
	reward_coef: float = 0.1
	value_coef: float = 0.1
	termination_coef: float = 1.0
	constraint_coef: float = 2.0
	entropy_coef: float = 1e-4
	horizon: int = 5
	batch_size: int = 256
	buffer_size: int = 500_000
	sequence_length: int = 50
	episode_length: int = 1000
	discount_denom: float = 5.0
	discount_min: float = 0.95
	discount_max: float = 0.995
	log_std_min: float = -10.0
	log_std_max: float = 2.0
	mpc: bool = False
	iterations: int = 6
	num_samples: int = 256
	num_elites: int = 32
	num_pi_trajs: int = 12
	temperature: float = 0.5
	min_std: float = 0.05
	max_std: float = 2.0
	compile: bool = False
	episodic: bool = True

	def __post_init__(self):
		prof = get_obs_profile(self.obs_profile)
		self.obs_shape = (prof.obs_dim,)
		self.bin_size = (self.vmax - self.vmin) / (self.num_bins - 1)
		if self.obs_profile == "dance":
			self.episode_length = 1565


def default_cfg(obs_profile: str = "dance", **overrides) -> SimpleNamespace:
	overrides.setdefault("obs_profile", obs_profile)
	init_fields = {k for k, f in KuavoTDMPC2Config.__dataclass_fields__.items() if f.init}
	cfg = KuavoTDMPC2Config(**{k: v for k, v in overrides.items() if k in init_fields})
	ns = SimpleNamespace(**cfg.__dict__)
	for k, v in overrides.items():
		if k not in KuavoTDMPC2Config.__dataclass_fields__:
			setattr(ns, k, v)
	return ns
