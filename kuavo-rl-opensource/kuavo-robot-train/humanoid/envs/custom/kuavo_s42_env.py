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
from humanoid.envs.custom.kuavo_s40_env import KuavoS40FreeEnv

from humanoid.utils.terrain import HumanoidTerrain
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.custom.kuavo_s40_config import KuavoS40Cfg, KuavoS40CfgPPO


class KuavoS42FreeEnv(KuavoS40FreeEnv):

    def __init__(self, cfg: KuavoS40Cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        scripts_path = cfg.env.scripts_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.is_ankle_pos_legal = torch.jit.load(os.path.join(scripts_path, "is_ankle_pos_legal.pt"))
        self.joint_to_motor_position = torch.jit.load(os.path.join(scripts_path, "joint_to_motor_position.pt"))
        self.get_joint_dumping_torque = torch.jit.load(os.path.join(scripts_path, "get_joint_dumping_torque.pt"))

    def check_termination(self):
        super().check_termination()
        ankle_illegal = (~self.is_ankle_pos_legal(self.dof_pos[:, 4:6])) | (~self.is_ankle_pos_legal(self.dof_pos[:, 10:12]))
        self.reset_buf |= ankle_illegal

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, [6, 12], 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 0., -1.)
        return torch.mean(reward, dim=1)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.kp_factors
        d_gains = self.d_gains * self.kd_factors

        p_torque = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
        motor_pos = self.joint_to_motor_position(self.dof_pos)
        d_torque = - self.get_joint_dumping_torque(self.dof_pos, motor_pos, d_gains, self.dof_vel)
        torques = p_torque + d_torque

        if self.cfg.domain_rand.randomize_motor_strength:
            motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0],
                                                      self.cfg.domain_rand.motor_strength_range[1],
                                                      (self.num_envs, self.num_actions), device=self.device)
            torques *= motor_strength_factors

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
