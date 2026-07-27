# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-1 stand-still env for TD-MPC2 (87-dim obs, zero velocity command, minimal DR)."""

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.utils import configclass

import ext_template.tasks.locomotion.velocity.mdp.rewards as local_rewards

from .flat_env_cfg import KuavoS42FlatEnvCfg


@configclass
class KuavoS42StandTDMPC2EnvCfg(KuavoS42FlatEnvCfg):
	"""WM curriculum stage 1: learn to stand upright with zero velocity command."""

	def __post_init__(self):
		super().__post_init__()
		self.scene.num_envs = 512

		# Zero velocity — only balance, no locomotion tracking
		self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
		self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
		self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
		self.commands.base_velocity.ranges.heading = (0.0, 0.0)

		# Standing-focused rewards
		self.rewards.track_lin_vel_xy_exp.weight = 0.0
		self.rewards.track_ang_vel_z_exp.weight = 0.0
		self.rewards.feet_air_time.weight = 0.0
		self.rewards.flat_orientation_l2.weight = -6.0
		self.rewards.base_height = RewTerm(
			func=local_rewards.base_height_l2,
			weight=-4.0,
			params={"target_height": 0.85},
		)
		self.rewards.stand_still_without_cmd.weight = -0.5
		self.rewards.gravity_aligned_when_stopping.weight = 0.2

		# Match PPO flat env action range (action=0 → default standing pose)
		self.actions.joint_pos.scale = 0.25

		# Disable domain randomization and pushes for stage 1
		self.events.physics_material = None
		self.events.add_base_mass = None
		self.events.scale_link_mass = None
		self.events.randomize_rigid_body_com = None
		self.events.scale_actuator_gains = None
		self.events.scale_joint_parameters = None
		self.events.base_external_force_torque = None

		# Nominal spawn: no initial velocity noise (untrained policy cannot recover from perturbations)
		self.events.reset_base.params = {
			"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)},
			"velocity_range": {
				"x": (0.0, 0.0),
				"y": (0.0, 0.0),
				"z": (0.0, 0.0),
				"roll": (0.0, 0.0),
				"pitch": (0.0, 0.0),
				"yaw": (0.0, 0.0),
			},
		}
		self.events.reset_robot_joints.params = {
			"position_range": (1.0, 1.0),
			"velocity_range": (0.0, 0.0),
		}

		# Termination tweaks for stand stage:
		# - dof_pos_illegal fires when S42 ankle TorchScript yields NaN (illegal
		#   joint combo). That kills the episode without a visible fall — looks
		#   like "moved a leg then vanished". Disable for stage-1; actuator still
		#   zeros NaN torques so sim continues.
		# - base_contact threshold 1N is tiny (clothing/mesh noise). Use 25N.
		self.terminations.dof_pos_illegal = None
		self.terminations.base_contact.params["threshold"] = 25.0


@configclass
class KuavoS42StandTDMPC2EnvCfg_PLAY(KuavoS42StandTDMPC2EnvCfg):
	def __post_init__(self):
		super().__post_init__()
		self.scene.num_envs = 1
		self.observations.policy.enable_corruption = False
		# Deterministic nominal spawn for video / eval
		self.events.reset_base.params = {
			"pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
			"velocity_range": {
				"x": (0.0, 0.0),
				"y": (0.0, 0.0),
				"z": (0.0, 0.0),
				"roll": (0.0, 0.0),
				"pitch": (0.0, 0.0),
				"yaw": (0.0, 0.0),
			},
		}
		self.events.reset_robot_joints.params = {
			"position_range": (1.0, 1.0),
			"velocity_range": (0.0, 0.0),
		}
