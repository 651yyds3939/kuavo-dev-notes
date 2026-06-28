# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
import os
from collections import deque

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from torch import nn
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import HumanoidTerrain
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from .fk import ForwardKinematics

torch.set_printoptions(precision=3, sci_mode=False)

class KuavoS40FreeEnv(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.fk = ForwardKinematics(mjcf_path=self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        self.fk.to(self.device)

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

        self.frame_stack_skip = self.cfg.env.frame_stack_skip
        self.frame_stack = self.cfg.env.frame_stack
        self.real_frame_stack = self.frame_stack // self.frame_stack_skip

        self.load_gait_model()
        self.build_period_history()
        self.contact_history = deque(maxlen=round(self.cfg.rewards.cycle_time / self.dt))
        for _ in range(self.contact_history.maxlen):
            self.contact_history.append(torch.zeros((self.num_envs, 2), device=self.device))
        self.vel_history = deque(maxlen=round(self.cfg.rewards.cycle_time / self.dt))
        for _ in range(self.vel_history.maxlen):
            self.vel_history.append(torch.zeros((self.num_envs, 3), device=self.device))

        self.compute_observations()

    def build_period_history(self):
        self.half_period_length = round(self.cfg.rewards.cycle_time / self.dt / 2)
        self.period_history = {}
        self.period_symmetric = self.cfg.rewards.period_symmetric
        for name in self.period_symmetric:
            self.period_history[name] = deque(maxlen=self.half_period_length)
            dim_num = self.period_symmetric[name]["dim_num"]
            for _ in range(self.half_period_length):
                self.period_history[name].append(torch.zeros((self.num_envs, dim_num), device=self.device))

    def load_gait_model(self):
        self.gait_model = nn.Sequential(
            nn.Linear(6, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU(),
            nn.Linear(64, 12 + 14 + 9)
        ).to(self.device)
        self.gait_model.load_state_dict(
            torch.load(self.cfg.env.gait_model_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)))

    def _push_robots(self):
        pass
        # """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        # """
        # max_vel = self.cfg.domain_rand.max_push_vel_xy
        # max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        # self.rand_push_force[:, :2] = torch_rand_float(
        #     -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        # self.root_states[:, 7:9] = self.rand_push_force[:, :2]
        #
        # self.rand_push_torque = torch_rand_float(
        #     -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)
        #
        # self.root_states[:, 10:13] = self.rand_push_torque
        #
        # self.gym.set_actor_root_state_tensor(
        #     self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        phase[self.commands[:, 4].to(bool)] = 0
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        # phase = self._get_phase()
        # sin_pos = torch.sin(2 * torch.pi * phase - 1.424)
        # # Add double support phase
        # stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # # left foot stance
        # stance_mask[:, 0] = sin_pos >= 0
        # # right foot stance
        # stance_mask[:, 1] = sin_pos < 0
        # # Double support phase
        # stance_mask[torch.abs(sin_pos) < 0.426] = 1

        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = self.ref_body_positions["leg_l6_link"][:, 2] < self.cfg.rewards.foot_height
        stance_mask[:, 1] = self.ref_body_positions["leg_r6_link"][:, 2] < self.cfg.rewards.foot_height
        stance_mask[self.commands[:, 4].to(bool)] = 1
        return stance_mask

    def get_manual_ref_dof_pos(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        ref_dof_pos[:, 2] = sin_pos_l * scale_1
        ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        ref_dof_pos[:, 8] = -sin_pos_r * scale_1
        ref_dof_pos[:, 9] = sin_pos_r * scale_2
        ref_dof_pos[:, 10] = -sin_pos_r * scale_1
        # Double support phase
        ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        ref_dof_pos[self.commands[:, 4].to(bool)] = self.default_dof_pos
        return ref_dof_pos

    def get_neural_ref_dof_pos(self):
        phase = self._get_phase()
        inputs = torch.zeros((self.num_envs, 6), device=self.device)
        inputs[:, 0] = torch.sin(2 * torch.pi * phase)
        inputs[:, 1] = torch.cos(2 * torch.pi * phase)
        inputs[:, 2] = self.cfg.rewards.cycle_time
        inputs[:, 3:6] = self.commands[:, :3]
        outputs = self.gait_model(inputs)

        self.ref_dof_pos = outputs[:, :self.num_dof]
        self.ref_euler_xy = outputs[:, self.num_dof:self.num_dof + 2]
        self.ref_height = outputs[:, self.num_dof + 2]
        self.ref_lin_vel = outputs[:, self.num_dof + 3:self.num_dof + 6]
        self.ref_ang_vel = outputs[:, self.num_dof + 6:self.num_dof + 9]

        self.ref_dof_pos[self.commands[:, 4].to(bool)] = self.default_dof_pos
        self.ref_euler_xy[self.commands[:, 4].to(bool), 0] = 0
        self.ref_euler_xy[self.commands[:, 4].to(bool), 1] = 0.05
        self.ref_height[self.commands[:, 4].to(bool)] = self.cfg.init_state.pos[-1]
        self.ref_lin_vel[self.commands[:, 4].to(bool)] = 0
        self.ref_ang_vel[self.commands[:, 4].to(bool)] = 0

    def get_ref_position_rotation(self):
        qpos = torch.zeros(self.num_envs, 7 + self.num_dofs, device=self.device)
        qpos[:, :2] = self.root_states[:, :2]
        qpos[:, 2] = self.ref_height
        qpos[:, 3:7] = quat_from_euler_xyz(self.ref_euler_xy[:, 0], self.ref_euler_xy[:, 1], self.base_euler_xyz[:, 2])
        qpos[:, 7:7 + self.num_dofs] = self.ref_dof_pos
        self.ref_body_positions, self.ref_body_rotations = self.fk(qpos, with_root=True)

    def compute_ref_state(self):
        self.get_neural_ref_dof_pos()
        self.get_ref_position_rotation()
        self.ref_action = 2 * self.ref_dof_pos

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 6] = 0.  # commands
        noise_vec[6: 32] = noise_scales.dof_pos * self.obs_scales.dof_pos
        if isinstance(noise_scales.dof_vel, float):
            noise_vec[32: 58] = noise_scales.dof_vel * self.obs_scales.dof_vel
        else:
            noise_vec[32: 58] = torch.tensor(noise_scales.dof_vel) * self.obs_scales.dof_vel
        noise_vec[58: 84] = noise_scales.last_torque
        noise_vec[84: 87] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[87: 89] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        noise_vec[89: 92] = noise_scales.lin_acc * self.obs_scales.lin_acc  # linear acc
        return noise_vec

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        return super().step(actions)

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3], self.commands[:, 4].unsqueeze(1)),
            dim=1)

        q = (self.dof_pos - self.default_dof_pos + self.joint_pos_bias) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.random_force_value[:, :5],  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
            self.com_displacement,
            self.restitution_coeffs,
            self.joint_pos_bias,
            self.joint_friction_coeffs,
            self.joint_armature_coeffs,
            self.kp_factors,
            self.kd_factors,
            self.ref_euler_xy,
            self.ref_height.reshape(-1, 1),
            self.ref_lin_vel,
            self.ref_ang_vel
        ), dim=-1)

        # print("Actual Vel:", self.base_lin_vel)

        base_euler_xy = self.base_euler_xyz.clone()[:, :2] + self.euler_xy_zero_pos
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,  # 12D
            dq,  # 12D
            # self.torques / self.torque_limits,  # 12D
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            base_euler_xy * self.obs_scales.quat,  # 3
            self.base_lin_acc * self.obs_scales.lin_acc,  # 3
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        obs_buf_all = obs_buf_all.reshape(self.num_envs, self.real_frame_stack, self.frame_stack_skip, -1)
        self.obs_buf = obs_buf_all[:, :, -1].reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

        for name in self.period_symmetric:
            self.period_history[name].append(self.get_period_symmetric_value(name).clone())
        self.contact_history.append(self.contact_forces[:, self.feet_indices, 2])

        self.vel_history.append(torch.cat([self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]], dim=-1))
        self.mean_vel = torch.mean(torch.stack([self.vel_history[i] for i in range(self.vel_history.maxlen)], dim=1), dim=1)

        if self.num_envs == 1:
            if self.episode_length_buf[0] == 0 and os.path.exists("play.txt"):
                os.remove("play.txt")
            with open("play.txt", "a") as f:
                f.write(",".join([str(s.item()) for s in self.root_states[0]]) + ",")
                f.write(",".join([str(s.item()) for s in self.dof_pos[0]]) + ",")
                f.write(",".join([str(s.item()) for s in self.dof_vel[0]]) + "\n")

    def get_period_symmetric_value(self, name):
        dim_num = self.period_symmetric[name]["dim_num"]
        if hasattr(self, name):
            return getattr(self, name)[:, :dim_num].clone()
        else:
            raise ValueError(f"Period symmetric name {name} not recognised")

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        self.lin_acc_filter.reset(env_ids)

    # @property
    # def standing_env_idx(self):
    #     return (torch.norm(self.commands[:, :3], dim=1) < 1e-3).nonzero(as_tuple=False).flatten()

    @property
    def rotating_env_idx(self):
        return (self.commands[:, 2].abs() > 1e-3).nonzero(as_tuple=False).flatten()

    def _resample_commands(self, env_ids, when_alive=False):
        super()._resample_commands(env_ids, when_alive)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.kp_factors
        d_gains = self.d_gains * self.kd_factors

        p_torque = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
        d_torque = - d_gains * self.dof_vel
        torques = p_torque + d_torque

        if self.cfg.domain_rand.randomize_motor_strength:
            motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_actions), device=self.device)
            torques *= motor_strength_factors

        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    # ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target

        # standing_vel_ratio = torch.exp(- 3 * torch.norm(self.base_lin_vel[self.standing_env_idx, :2], dim=1))
        # diff[self.standing_env_idx, :12] *= standing_vel_ratio.reshape(-1, 1)
        sigma = self.cfg.rewards.joint_pos_sigma * torch.ones(self.num_envs, device=self.device)
        # diff[(self.contact_forces[:, 6, 2] > 5) & (~self.commands[:, 4].bool()), 3:6] /= 2
        # diff[(self.contact_forces[:, 12, 2] > 5) & (~self.commands[:, 4].bool()), 9:12] /= 2
        # rew = torch.exp(-sigma * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1)
        for idx in [0, 5, 6, 11]:
            diff[~self.commands[:, 4].bool(), idx] *= 0
        diff[~self.commands[:, 4].bool(), 2] *= 0.5
        diff[~self.commands[:, 4].bool(), 8] *= 0.5
        rew = torch.exp(-sigma * torch.norm(diff, dim=1))
        # print(pos_target[0, :6], joint_pos[0, :6])
        # print(diff[0, :12], rew[0].item())
        rew[self.is_pushing] = 1
        ratio = torch.clip(self.resample_cmd_length_buf * self.dt / self.cfg.rewards.cycle_time, 0, 1)
        rew = rew * ratio + (1 - ratio)
        return rew

    def _reward_foot_pos(self):
        stance_mask = self.contact_forces[:, self.feet_indices, 2] > 100.
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1) - self.cfg.rewards.foot_height
        measured_heights[torch.isnan(measured_heights)] = 0

        left_pos_diff = self.rigid_state[:, 6, :3] - self.ref_body_positions["leg_l6_link"]
        left_pos_diff[:, 2] -= measured_heights
        # left_quat_diff = torch.acos((self.rigid_state[:, 6, 3:7] * self.body_rotations["leg_l6_link"]).sum(-1))
        right_pos_diff = self.rigid_state[:, 12, :3] - self.ref_body_positions["leg_r6_link"]
        right_pos_diff[:, 2] -= measured_heights
        # right_quat_diff = torch.acos((self.rigid_state[:, 12, 3:7] * self.body_rotations["leg_r6_link"]).sum(-1))
        # left_quat_diff = (self.rigid_state[:, 6, 3:7] * self.body_rotations["leg_l6_link"]).sum(-1)
        # right_quat_diff = (self.rigid_state[:, 12, 3:7] * self.body_rotations["leg_r6_link"]).sum(-1)
        # left_pos_diff[:, 2] *= 3
        # right_pos_diff[:, 2] *= 3
        rew = torch.exp(-torch.norm(torch.cat([left_pos_diff, right_pos_diff], dim=-1), dim=1) * 20)
        rew[self.is_pushing] = 1
        return rew

    def _reward_marking_time(self):
        mt_ids = (torch.norm(self.commands[:, :3], dim=-1) < 1e-3) & (~self.commands[:, 4].to(bool))
        mt_ids &= (self.contact_forces[:, self.feet_indices[0], 2] > 5)
        mt_ids &= (self.contact_forces[:, self.feet_indices[1], 2] > 5)

        pos_l = quat_rotate_inverse(self.base_quat, self.rigid_state[:, self.feet_indices[0], :3] - self.root_states[:, :3])
        pos_r = quat_rotate_inverse(self.base_quat, self.rigid_state[:, self.feet_indices[1], :3] - self.root_states[:, :3])

        # print(touch_ids, torch.norm(self.dof_pos[:, [2, 3, 4]] - self.dof_pos[:, [8, 9, 10]], dim=-1))

        rew = torch.zeros(self.num_envs, device=self.device)
        feet_x_rew = torch.exp(-(pos_r[:, 0] - pos_l[:, 0]).abs() * 100)
        feet_y_rew = torch.exp(-(pos_r[:, 1] + pos_l[:, 1]).abs() * 100)
        dof_pos_rew = torch.exp(-torch.norm(self.dof_pos[:, 2:5] - self.dof_pos[:, 8:11], dim=-1) * 30)
        rew[mt_ids] += (feet_x_rew[mt_ids] + feet_y_rew[mt_ids] + dof_pos_rew[mt_ids]) / 3
        return rew

    def _reward_half_period(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        max_reward = 0
        for name in self.period_symmetric:
            target = self.period_history[name][0].clone()
            target[self.commands[:, 4].to(bool)] = self.get_period_symmetric_value(name)[self.commands[:, 4].to(bool)].clone()
            target[:, :12] = torch.roll(target[:, :12], shifts=6, dims=1)
            target[:, 12:] = torch.roll(target[:, 12:], shifts=7, dims=1)
            target[:, [0, 1, 5, 6, 7, 11]] *= -1
            target[:, [13, 14, 16, 18, 20, 21, 23, 25]] *= -1

            diff = self.get_period_symmetric_value(name) - target
            # for idx in [1, 7]:
            #     diff[self.rotating_env_idx, idx] = 0
            diff[:, [0, 1, 5, 6, 7, 11]] *= 5
            sigma = self.period_symmetric[name]["sigma"] * torch.ones(self.num_envs, device=self.device)
            sigma[~(self.commands[:, 1:] == 0).all(dim=1)] = 0
            # sigma[self.commands[:, 4].to(bool)] *= 2.
            rew = torch.exp(-sigma * torch.norm(diff, dim=1))
            # print(name, rew)
            # if name == "dof_pos":
            #     print(name, diff[:, :6],  torch.norm(diff, dim=1), rew)
            # if name == "dof_pos":
            #     print(diff[:, :12])
            #     print("now", self.get_period_symmetric_value(name))
            #     print("target", target)
            # print(name, torch.exp(-half_period_penalty * sigma))
            # with open ("reward.txt", "a") as f:
            #     if name == "dof_pos":
            #         f.write(f"{torch.exp(-half_period_penalty * sigma).item()}\n")
            reward += rew * self.period_symmetric[name]["scale"]
            max_reward += self.period_symmetric[name]["scale"]
        # ratio = torch.clip(self.resample_cmd_length_buf * self.dt / self.cfg.rewards.cycle_time, 0, 1)
        # reward = reward * ratio + max_reward * (1 - ratio)
        reward[self.is_pushing] = max_reward
        return reward


    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        rew = (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
        # print(rew, d_min, d_max)
        # print(self.actions[:, [0, 6]])
        return rew

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        # gravity_proj = [quat_rotate_inverse(self.rigid_state[:, idx, 3:7], self.gravity_vec)[:, 2] for idx in self.feet_indices]
        # gravity_proj = torch.stack(gravity_proj, dim=1) ** 4

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        # rew *= gravity_proj
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, [6, 12], 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)


    def _reward_feet_contact_same(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contacts = torch.stack([self.contact_history[i] for i in range(len(self.contact_history))], dim=1)
        mean_force = contacts.mean(dim=1)
        diff1 = (mean_force[:, 0] - mean_force[:, 1])
        diff2 = (contacts[:, -1] - contacts[:, -self.contact_history.maxlen // 2]).sum(-1)
        rew1 = torch.exp(- 0.01 * diff1.abs())
        rew2 = torch.exp(- 0.01 * diff2.abs())
        return (rew1 + rew2) / 2

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        # quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2] - self.ref_euler_xy), dim=1) * 10)
        # # orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        # return quat_mismatch
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # Attention: no friction force here
        contact_force = self.contact_forces[:, self.feet_indices, 2]
        rew = (contact_force.sum(-1) - self.cfg.rewards.max_contact_force).clip(0, 400)
        rew[self.episode_length_buf < 20] = 0
        return rew

    # def _reward_default_joint_pos(self):
    #     """
    #     Calculates the reward for keeping joint positions close to default positions, with a focus
    #     on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
    #     """
    #     joint_diff = self.dof_pos - self.default_joint_pd_target
    #     left_yaw_roll = joint_diff[:, :2]
    #     right_yaw_roll = joint_diff[:, 6: 8]
    #     yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    #     yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    #     return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    #
    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """

        stance_mask = self.contact_forces[:, self.feet_indices, 2] > 100.
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1) - self.cfg.rewards.foot_height
        measured_heights[torch.isnan(measured_heights)] = 0
        base_height = self.root_states[:, 2] - measured_heights
        rew = torch.exp(-torch.abs(base_height - self.ref_height) * 20)
        return rew

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_x_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        x_vel_error = torch.abs(self.commands[:, 0] - self.mean_vel[:, 0])
        rew = torch.zeros(self.num_envs, device=self.device)
        for sigma in self.cfg.rewards.x_tracking_sigmas:
            rew += torch.exp(-sigma * x_vel_error) / len(self.cfg.rewards.x_tracking_sigmas)
        ref_instant_vel = self.commands[:, 0]
        rew += torch.exp(-10 * torch.square(ref_instant_vel - self.base_lin_vel[:, 0]))
        # print("x", torch.exp(-10 * torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])))
        # rew += torch.exp(-10 * torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]))
        # rew[self.is_pushing] = 2
        # print("x:", self.ewm_x_vel.item(), x_vel_error.item(), rew.item())
        # ratio = torch.clip(self.resample_cmd_length_buf * self.dt / self.cfg.rewards.cycle_time, 0, 1)
        # rew = rew * ratio + (1 - ratio)
        #     f.write(f"{self.mean_vel[:, 0].item()},{self.mean_vel[:, 1].item()},{self.mean_vel[:, 2].item()}\n")
        return rew / 2


    def _reward_tracking_y_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        y_vel_error = torch.abs(self.commands[:, 1] - self.mean_vel[:, 1])
        rew = torch.zeros(self.num_envs, device=self.device)
        for sigma in self.cfg.rewards.y_tracking_sigmas:
            rew += torch.exp(-sigma * y_vel_error) / len(self.cfg.rewards.y_tracking_sigmas)
        ref_instant_vel = self.commands[:, 1]
        rew += torch.exp(-10 * torch.square(ref_instant_vel - self.base_lin_vel[:, 1]))
        # print("y", torch.exp(-10 * torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])))
        # rew += torch.exp(-10 * torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]))
        # rew[self.is_pushing] = 2
        # print("y:", self.ewm_y_vel.item(), y_vel_error.item(), rew.item())
        # ratio = torch.clip(self.resample_cmd_length_buf * self.dt / self.cfg.rewards.cycle_time, 0, 1)
        # rew = rew * ratio + (1 - ratio)
        return rew / 2

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        ang_vel_error = torch.abs(self.commands[:, 2] - self.mean_vel[:, 2])
        rew = torch.zeros(self.num_envs, device=self.device)
        for sigma in self.cfg.rewards.yaw_tracking_sigmas:
            rew += torch.exp(-sigma * ang_vel_error) / len(self.cfg.rewards.yaw_tracking_sigmas)
        # ref_instant_vel = 0.5 * self.commands[:, 2] + 0.5 * self.ref_ang_vel[:, 2]
        ref_instant_vel = self.commands[:, 2]
        rew += torch.exp(-10 * torch.square(ref_instant_vel - self.base_ang_vel[:, 2]))
        # print("yaw", torch.exp(-10 * torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])))
        # rew += torch.exp(-10 * torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2]))
        # rew[self.is_pushing] = 2
        # print("yaw:", self.ewm_yaw_vel.item(), ang_vel_error.item(), rew.item())
        # ratio = torch.clip(self.resample_cmd_length_buf * self.dt / self.cfg.rewards.cycle_time, 0, 1)
        # rew = rew * ratio + (1 - ratio)
        return rew / 2

    # def _reward_feet_clearance(self):
    #     """
    #     Calculates reward based on the clearance of the swing leg from the ground during movement.
    #     Encourages appropriate lift of the feet during the swing phase of the gait.
    #     """
    #     # Compute feet contact mask
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    #
    #     # Get the z-position of the feet and compute the change in z-position
    #     feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
    #     delta_z = feet_z - self.last_feet_z
    #     self.feet_height += delta_z
    #     self.last_feet_z = feet_z
    #
    #     # Compute swing mask
    #     swing_mask = 1 - self._get_gait_phase()
    #
    #     # feet height should be closed to target feet height at the peak
    #     rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
    #     rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    #     self.feet_height *= ~contact
    #     return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        weight = torch.tensor([1, 1, 1, 1, 2, 3] * 2 + [1] * 14, device=self.device)
        rew = torch.sum(torch.square(self.torques * weight), dim=1)
        rew[self.is_pushing] /= 3
        return rew

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        weight = torch.tensor([3, 3, 1, 1, 1, 3] * 2 + [1] * 14, device=self.device)
        rew = torch.sum(torch.square(self.dof_vel * weight), dim=1)
        return rew

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_dof_jerk(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
        last_dof_acc = (self.last_dof_vel - self.last_last_dof_vel) / self.dt
        return torch.sum((torch.square(dof_acc - last_dof_acc) / self.dt), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

class KuavoS40SKFreeEnv(KuavoS40FreeEnv):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def manual_style(self, outputs):
        cycle_signal = 2 * torch.pi * self._get_phase()
        cycle_signal %= 2 * torch.pi
        abs_x_vel = torch.clip(torch.abs(self.commands[:, 0]), 0, 0.3)
        clipped_x_vel = torch.clip(self.commands[:, 0], 0, 0.3)
        left_offset = torch.where(
            (5 * torch.pi / 4 < cycle_signal) & (cycle_signal < 2 * torch.pi),
            torch.sin(4 / 3 * (cycle_signal - 5 * torch.pi / 4)),
            0.0
        )
        right_offset = torch.where(
            (torch.pi / 4 < cycle_signal) & (cycle_signal < torch.pi),
            torch.sin(4 / 3 * (cycle_signal - torch.pi / 4)),
            0.0
        )
        
        outputs[:, 2] += 0.8 * abs_x_vel - left_offset * 0.5 * clipped_x_vel
        outputs[:, 3] += -1.5 * abs_x_vel  + left_offset * 1 * clipped_x_vel
        outputs[:, 4] += 0.8 * abs_x_vel   + left_offset * 1.5 * clipped_x_vel
        outputs[:, 8] += 0.8 * abs_x_vel - right_offset * 0.5 * clipped_x_vel
        outputs[:, 9] += -1.6 * abs_x_vel  + right_offset * 1 * clipped_x_vel
        outputs[:, 10] += 0.8 * abs_x_vel  + right_offset * 1.5 * clipped_x_vel
        outputs[:, 12:self.num_dof] = 0
        outputs[:, 12] = torch.cos(cycle_signal) * 0.2
        outputs[:, 15] = torch.clip(torch.cos(cycle_signal), -1, 0) * 0.2
        outputs[:, 19] = - torch.cos(cycle_signal) * 0.2
        outputs[:, 22] = torch.clip(- torch.cos(cycle_signal), -1, 0) * 0.2
        outputs[:, self.num_dof + 1] += -0.05 * abs_x_vel
        outputs[:, self.num_dof + 2] += 0.12 * torch.clip(torch.abs(self.commands[:, 0]), 0, 0.4)
        return outputs

    def get_neural_ref_dof_pos(self):
        phase = self._get_phase()
        inputs = torch.zeros((self.num_envs, 6), device=self.device)
        inputs[:, 0] = torch.sin(2 * torch.pi * phase)
        inputs[:, 1] = torch.cos(2 * torch.pi * phase)
        inputs[:, 2] = self.cfg.rewards.cycle_time
        inputs[:, 3:6] = self.commands[:, :3]

        outputs = self.gait_model(inputs)
        outputs = self.manual_style(outputs)

        self.ref_dof_pos = outputs[:, :self.num_dof]
        self.ref_euler_xy = outputs[:, self.num_dof:self.num_dof + 2]
        self.ref_height = outputs[:, self.num_dof + 2]
        self.ref_lin_vel = outputs[:, self.num_dof + 3:self.num_dof + 6]
        self.ref_ang_vel = outputs[:, self.num_dof + 6:self.num_dof + 9]

        self.ref_dof_pos[self.commands[:, 4].to(bool)] = self.default_dof_pos
        self.ref_euler_xy[self.commands[:, 4].to(bool), 0] = 0
        self.ref_euler_xy[self.commands[:, 4].to(bool), 1] = 0.05
        self.ref_height[self.commands[:, 4].to(bool)] = self.cfg.init_state.pos[-1]
        self.ref_lin_vel[self.commands[:, 4].to(bool)] = 0
        self.ref_ang_vel[self.commands[:, 4].to(bool)] = 0


class KuavoS40AmassFreeEnv(KuavoS40FreeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_period_history(self):
        max_period = self.get_expected_period_length(torch.tensor([self.cfg.commands.min_lin_vel])).item()
        self.half_period_length = round(max_period / self.dt / 2)
        self.period_history = {}
        self.period_symmetric = self.cfg.rewards.period_symmetric
        for name in self.period_symmetric:
            self.period_history[name] = deque(maxlen=self.half_period_length)
            dim_num = self.period_symmetric[name]["dim_num"]
            for _ in range(self.half_period_length):
                self.period_history[name].append(torch.zeros((self.num_envs, dim_num), device=self.device))

    def _init_buffers(self):
        super()._init_buffers()
        self.cycle_time_buffer = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def post_physics_step(self):
        super().post_physics_step()
        self.cycle_time_buffer += 1

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.cycle_time_buffer[env_ids] = 0

    def get_expected_period_length(self, x_command_vel):
        length = torch.zeros(x_command_vel.shape[0], device=self.device)
        length[x_command_vel > 0.] = 1. / (0.56 * x_command_vel[x_command_vel > 0.] + 0.39)
        return length

    def get_neural_ref_dof_pos(self):
        phase = self._get_phase()
        inputs = torch.zeros((self.num_envs, 6), device=self.device)
        inputs[:, 0] = torch.sin(2 * torch.pi * phase)
        inputs[:, 1] = torch.cos(2 * torch.pi * phase)
        inputs[:, 2] = self.get_expected_period_length(self.commands[:, 0])
        inputs[:, 3:6] = self.commands[:, :3]
        ref_dof_pos = self.gait_model(inputs)
        ref_dof_pos[torch.norm(self.commands[:, :3], dim=1) < 0.1] = self.default_dof_pos
        return ref_dof_pos

    def _resample_commands(self, env_ids, when_alive=False):
        pre_period_length = (self.get_expected_period_length(self.commands[env_ids, 0]) / self.dt).long()
        super()._resample_commands(env_ids, when_alive=when_alive)

        standing_env_ids = env_ids[torch.norm(self.commands[env_ids, :3], dim=1) < 1e-3]
        self.commands[standing_env_ids, 4] = 1

        if when_alive and len(env_ids) > 0:
            period_length = (self.get_expected_period_length(self.commands[env_ids, 0]) / self.dt).long()
            self.cycle_time_buffer[env_ids] = ((self.cycle_time_buffer[env_ids] % pre_period_length) * period_length / pre_period_length).long()

    def _get_phase(self):
        cycle_time = self.get_expected_period_length(self.commands[:, 0])
        phase = torch.zeros(self.num_envs, device=self.device)
        phase[cycle_time > 0] = self.cycle_time_buffer[cycle_time > 0] * self.dt / cycle_time[cycle_time > 0]
        return phase

    def _reward_half_period(self):
        reward = torch.zeros(self.num_envs, device=self.device)

        cycle_time = self.get_expected_period_length(self.commands[:, 0])
        half_period_length = (cycle_time / self.dt / 2).to(torch.int)

        for name in self.period_symmetric:
            dim_num = self.period_symmetric[name]["dim_num"]
            target = torch.zeros(self.num_envs, dim_num, device=self.device)
            for i, value in enumerate(torch.unique(half_period_length)):
                target[half_period_length == value] = self.period_history[name][-value][half_period_length == value]
            target[:, :12] = torch.roll(target[:, :12], shifts=6, dims=1)
            target[:, 12:] = torch.roll(target[:, 12:], shifts=7, dims=1)
            target[:, [0, 1, 5, 6, 7, 11]] *= -1
            target[:, [13, 14, 16, 18, 20, 21, 23, 25]] *= -1

            diff = self.get_period_symmetric_value(name) - target
            for idx in [1, 7]:
                diff[self.rotating_env_idx, idx] = 0

            half_period_penalty = torch.sum(torch.square(diff), dim=1)
            sigma = self.period_symmetric[name]["sigma"]
            # print(name, torch.exp(-half_period_penalty * sigma))
            reward += torch.exp(-half_period_penalty * sigma)

        return reward
