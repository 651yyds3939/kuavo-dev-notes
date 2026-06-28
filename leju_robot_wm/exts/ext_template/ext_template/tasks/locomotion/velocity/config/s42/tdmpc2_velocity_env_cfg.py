# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Flat velocity-tracking env for TD-MPC2 (87-dim policy obs, no dance reference)."""

import math

from omni.isaac.lab.utils import configclass

from .flat_env_cfg import KuavoS42FlatEnvCfg


@configclass
class KuavoS42VelocityTDMPC2EnvCfg(KuavoS42FlatEnvCfg):
	"""S42/S46 velocity tracking with non-zero command ranges for world-model training."""

	def __post_init__(self):
		super().__post_init__()
		# 8GB VRAM: lower default than PPO 8192
		self.scene.num_envs = 512
		self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
		self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
		self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
		self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
		self.rewards.track_lin_vel_xy_exp.weight = 1.0
		self.rewards.track_ang_vel_z_exp.weight = 0.5


@configclass
class KuavoS42VelocityTDMPC2EnvCfg_PLAY(KuavoS42VelocityTDMPC2EnvCfg):
	def __post_init__(self):
		super().__post_init__()
		self.scene.num_envs = 16
		self.observations.policy.enable_corruption = False
		self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
		self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
		self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
