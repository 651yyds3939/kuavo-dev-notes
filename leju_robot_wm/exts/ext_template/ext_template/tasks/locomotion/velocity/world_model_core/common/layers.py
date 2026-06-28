"""Neural building blocks for Kuavo state-based TD-MPC2 (no pixel / Gymnasium paths)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy

from .. import helper


class Ensemble(nn.Module):
	"""Vectorized ensemble of modules."""

	def __init__(self, modules, **kwargs):
		super().__init__()
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f"Vectorized {len(self)}x " + self._repr


class SimNorm(nn.Module):
	"""Simplicial normalization."""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""Linear layer with LayerNorm, activation, and optionally dropout."""

	def __init__(self, *args, dropout=0.0, act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return (
			f"NormedLinear(in_features={self.in_features}, "
			f"out_features={self.out_features}, "
			f"bias={self.bias is not None}{repr_dropout}, "
			f"act={self.act.__class__.__name__})"
		)


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
	"""MLP with LayerNorm, Mish activations, and optionally dropout."""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp_layers = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp_layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
	mlp_layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp_layers)


class StateJointEmbedding(nn.Module):
	"""
	Joint-embedding MLP encoder for 115-dim Kuavo proprioceptive state vectors.
	Replaces upstream pixel / dict-based encoders.
	"""

	def __init__(self, cfg):
		super().__init__()
		in_dim = cfg.obs_shape[0]
		hidden = max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim]
		self.net = mlp(in_dim, hidden, cfg.latent_dim, act=SimNorm(cfg))

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		if obs.ndim == 3:
			b, t, d = obs.shape
			return self.net(obs.reshape(b * t, d)).reshape(b, t, -1)
		return self.net(obs)


def state_encoder(cfg) -> StateJointEmbedding:
	"""Build the Kuavo state encoder (obs_dim -> latent)."""
	return StateJointEmbedding(cfg)


def api_model_conversion(target_state_dict, source_state_dict):
	"""Converts a checkpoint from the old API to the torch.compile compatible API."""
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ["weight", "bias", "ln.weight", "ln.bias"]
	new_state_dict = dict()

	for key, val in list(source_state_dict.items()):
		if key.startswith("_Qs."):
			num = key[len("_Qs.params.") :]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith("_target_Qs."):
			num = key[len("_target_Qs.params.") :]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	for prefix in ("_Qs.", "_detach_Qs_", "_target_Qs_"):
		for key in ("__batch_size", "__device"):
			new_key = prefix + "params." + key
			new_state_dict[new_key] = target_state_dict[new_key]

	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	for key in target_state_dict.keys():
		if "Qs" in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	for key in source_state_dict.keys():
		assert "Qs" not in key, f"key {key} contains 'Qs'"

	new_state_dict["log_std_min"] = target_state_dict["log_std_min"]
	new_state_dict["log_std_dif"] = target_state_dict["log_std_dif"]
	source_state_dict.update(new_state_dict)
	return source_state_dict
