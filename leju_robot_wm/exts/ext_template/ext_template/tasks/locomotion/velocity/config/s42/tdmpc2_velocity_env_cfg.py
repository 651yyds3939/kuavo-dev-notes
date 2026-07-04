# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Flat velocity-tracking env for TD-MPC2 (87-dim policy obs, no dance reference)."""

import math

from omni.isaac.lab.utils import configclass

from .flat_env_cfg import KuavoS42FlatEnvCfg


@configclass
class KuavoS42VelocityTDMPC2EnvCfg(KuavoS42FlatEnvCfg):
	"""WM curriculum stage 2: velocity tracking with moderate resets and light DR."""

	def __post_init__(self):
		super().__post_init__()
		self.scene.num_envs = 512
		self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
		self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
		self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
		self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
		self.rewards.track_lin_vel_xy_exp.weight = 1.0
		self.rewards.track_ang_vel_z_exp.weight = 0.5
		self.actions.joint_pos.scale = 0.20

		# Moderate reset (avoid rough_env 0.5–1.5 joint scale that breaks balance)
		self.events.reset_base.params = {
			"pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.3, 0.3)},
			"velocity_range": {
				"x": (-0.1, 0.1),
				"y": (-0.1, 0.1),
				"z": (-0.1, 0.1),
				"roll": (-0.1, 0.1),
				"pitch": (-0.1, 0.1),
				"yaw": (-0.1, 0.1),
			},
		}
		self.events.reset_robot_joints.params = {
			"position_range": (0.95, 1.05),
			"velocity_range": (0.0, 0.0),
		}


@configclass
class KuavoS42VelocityTDMPC2EnvCfg_PLAY(KuavoS42VelocityTDMPC2EnvCfg):
	def __post_init__(self):
		super().__post_init__()
		self.scene.num_envs = 1
		self.observations.policy.enable_corruption = False
		self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
		self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
		self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
		# Play: no domain randomization or pushes
		self.events.base_external_force_torque = None
		self.events.randomize_rigid_body_com = None
		self.events.physics_material = None
		self.events.add_base_mass = None
		self.events.scale_actuator_gains = None
		self.events.scale_link_mass = None
		self.events.scale_joint_parameters = None
