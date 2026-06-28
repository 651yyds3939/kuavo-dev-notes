"""
Kuavo S49 TD-MPC2 agent — state-based (115-dim obs, 26-dim action).

Stripped of upstream Gymnasium / pixel-encoder paths. Includes a Constraint/Termination
decoder that penalizes base_contact and fallen (upright_factor < 0.75) behaviours
during both training and MPPI latent imagination.
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from . import helper
from .common.layers import api_model_conversion
from .common.world_model import WorldModel


class TDMPC2(torch.nn.Module):
	"""TD-MPC2 agent for Kuavo S49 dance imitation (Model-Based world model + MPPI)."""

	def __init__(self, cfg=None):
		super().__init__()
		self.cfg = cfg or helper.default_cfg()

		self.device = torch.device("cuda:0")
		self.model = WorldModel(self.cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{"params": self.model._encoder.parameters(), "lr": self.cfg.lr * self.cfg.enc_lr_scale},
			{"params": self.model._dynamics.parameters()},
			{"params": self.model._reward.parameters()},
			{"params": self.model._termination.parameters()},
			{"params": self.model._constraint.parameters()},
			{"params": self.model._Qs.parameters()},
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = helper.RunningScale(self.cfg).to(self.device)
		self.cfg.iterations += 2 * int(self.cfg.action_dim >= 20)
		self.discount = self._get_discount(self.cfg.episode_length)
		print("Episode length:", self.cfg.episode_length)
		print("Discount factor:", self.discount)
		self.register_buffer(
			"_prev_mean",
			torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device),
		)
		if getattr(self.cfg, "compile", False):
			print("Compiling update function with torch.compile...")
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if getattr(self.cfg, "compile", False):
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		frac = episode_length / self.cfg.discount_denom
		return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, use_mpc=None):
		"""Select action for a single observation [obs_dim]."""
		return self.act_batch(obs.unsqueeze(0), t0=t0, eval_mode=eval_mode, use_mpc=use_mpc)[0]

	@torch.no_grad()
	def act_batch(self, obs, t0=False, eval_mode=False, use_mpc=None):
		"""Select actions for batched observations [batch, obs_dim]."""
		use_mpc = self.cfg.mpc if use_mpc is None else use_mpc
		obs = obs.to(self.device, non_blocking=True)
		if use_mpc and obs.shape[0] == 1:
			return self.plan(obs, t0=t0, eval_mode=eval_mode).unsqueeze(0)
		z = self.model.encode(obs)
		action, info = self.model.pi(z)
		if eval_mode:
			action = info["mean"]
		return action.clamp(-1, 1)

	@torch.no_grad()
	def _estimate_value(self, z, actions):
		"""
		Estimate trajectory value in latent imagination.
		Applies heavy penalty when constraint decoder predicts fallen / base_contact.
		"""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		constraint = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = helper.two_hot_inv(self.model.reward(z, actions[t]), self.cfg)
			z = self.model.next(z, actions[t])
			G = G + discount * (1 - termination) * (1 - constraint) * reward
			discount = discount * self.discount
			termination = torch.clip(termination + (self.model.termination(z) > 0.5).float(), max=1.0)
			constraint = torch.clip(constraint + (self.model.constraint(z) > 0.5).float(), max=1.0)
			G = G - discount * helper.CONSTRAINT_PENALTY * constraint
		action, _ = self.model.pi(z)
		return G + discount * (1 - termination) * (1 - constraint) * self.model.Q(z, action, return_type="avg")

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False):
		z = self.model.encode(obs)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device
			)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t], _ = self.model.pi(_z)
				_z = self.model.next(_z, pi_actions[t])
			pi_actions[-1], _ = self.model.pi(_z)

		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full(
			(self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device
		)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, : self.cfg.num_pi_trajs] = pi_actions

		for _ in range(self.cfg.iterations):
			r = torch.randn(
				self.cfg.horizon,
				self.cfg.num_samples - self.cfg.num_pi_trajs,
				self.cfg.action_dim,
				device=std.device,
			)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs :] = actions_sample

			value = self._estimate_value(z, actions).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = (
				(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
				/ (score.sum(0) + 1e-9)
			).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)

		rand_idx = helper.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs):
		action, info = self.model.pi(zs)
		qs = self.model.Q(zs, action, return_type="avg", detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)
		return TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, step_mask):
		action, _ = self.model.pi(next_z)
		return reward + self.discount * (1 - terminated) * step_mask * self.model.Q(
			next_z, action, return_type="min", target=True
		)

	def _update(self, obs, action, reward, terminated, constraint=None, step_mask=None):
		"""
		World-model update with constraint BCE loss (anti-躺平 dam).

		Args:
			obs: ``[horizon+1, batch, 115]``
			action: ``[horizon, batch, 26]``
			reward / terminated / constraint: ``[horizon, batch, 1]``
			step_mask: validity mask for padded timesteps
		"""
		if step_mask is None:
			step_mask = torch.ones_like(reward)
		if constraint is None:
			constraint = helper.constraint_label_from_obs(obs[1:])

		with torch.no_grad():
			next_z = self.model.encode(obs[1:])
			td_targets = self._td_target(next_z, reward, terminated, step_mask)

		self.model.train()

		zs = torch.empty(self.cfg.horizon + 1, obs.shape[1], self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0])
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action)
			m = step_mask[t]
			consistency_loss = consistency_loss + (F.mse_loss(z, _next_z, reduction="none").mean(-1, keepdim=True) * m).mean() * self.cfg.rho ** t
			zs[t + 1] = z

		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, return_type="all")
		reward_preds = self.model.reward(_zs, action)
		termination_pred = self.model.termination(zs[1:], unnormalized=True)
		constraint_pred = self.model.constraint(zs[1:], unnormalized=True)

		reward_loss, value_loss = 0, 0
		for t, (rew_pred, rew, td_t, qs_t) in enumerate(
			zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
		):
			m = step_mask[t]
			reward_loss = reward_loss + (helper.soft_ce(rew_pred, rew, self.cfg) * m).mean() * self.cfg.rho ** t
			for qs_unbind in qs_t.unbind(0):
				value_loss = value_loss + (helper.soft_ce(qs_unbind, td_t, self.cfg) * m).mean() * self.cfg.rho ** t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		termination_loss = (
			F.binary_cross_entropy_with_logits(termination_pred, terminated, reduction="none") * step_mask
		).mean()
		constraint_loss = (
			F.binary_cross_entropy_with_logits(constraint_pred, constraint, reduction="none") * step_mask
		).mean()
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.termination_coef * termination_loss
			+ self.cfg.constraint_coef * constraint_loss
			+ self.cfg.value_coef * value_loss
		)

		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		pi_info = self.update_pi(zs.detach())
		self.model.soft_update_target_Q()
		self.model.eval()

		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"constraint_loss": constraint_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		info.update(helper.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(helper.termination_statistics(torch.sigmoid(constraint_pred[-1]), constraint[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""Main update: sample from SequenceBuffer (CPU -> GPU) and train world model."""
		obs, action, reward, terminated, constraint, mask = buffer.sample_horizon(
			self.cfg.horizon, self.cfg.batch_size
		)
		if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
			torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, constraint, mask)
