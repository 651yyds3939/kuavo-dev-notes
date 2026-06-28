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

from itertools import product

from humanoid.envs.custom.kuavo_s40_config import KuavoS40Cfg, KuavoS40CfgPPO


class KuavoS42Cfg(KuavoS40Cfg):
    """
    Configuration class for the XBotL humanoid robot.
    """

    class env(KuavoS40Cfg.env):
        gait_model_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/gait_walk102.pth'
        scripts_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/scripts'

    class asset(KuavoS40Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/mjcf/biped_s42.xml'

    class init_state(KuavoS40Cfg.init_state):
        pos = [0., 0., 0.85]
        default_joint_angles = {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [0.0, 0.0, -0.46, 0.84, -0.44, 0.0]):
                default_joint_angles[f'leg_{lr}{idx}_joint'] = value
        for lr, idx in product(['l', 'r'], range(1, 8)):
            default_joint_angles[f'zarm_{lr}{idx}_joint'] = 0.

    class commands(KuavoS40Cfg.commands):
        class ranges(KuavoS40Cfg.commands.ranges):
            lin_vel_x = [-0.4, 1.0]  # min max [m/s]
            lin_vel_y = [-0.2, 0.2]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]

    class rewards(KuavoS40Cfg.rewards):
        foot_height = 0.07
        max_contact_force = 600
        cycle_time = 1.02

    class control(KuavoS40Cfg.control):
        # PD Drive parameters:
        stiffness, damping = {}, {}

        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [60.0, 60.0, 60.0, 60.0, 30.0, 15.0]):
                stiffness[f'leg_{lr}{idx}_joint'] = value
        for lr, idx in product(['l', 'r'], range(1, 8)):
            stiffness[f'zarm_{lr}{idx}_joint'] = 15.0

        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [10.0, 6.0, 12.0, 12.0, 22.0, 22.0]):
                damping[f'leg_{lr}{idx}_joint'] = value
        for lr, idx in product(['l', 'r'], range(1, 8)):
            damping[f'zarm_{lr}{idx}_joint'] = 3


class KuavoS42LSCfg(KuavoS42Cfg):
    class env(KuavoS42Cfg.env):
        frame_stack = 100
        frame_stack_skip = 1
        num_single_obs = 47 + 3 + 14 * 3
        num_observations = int(frame_stack // frame_stack_skip * num_single_obs)


class KuavoS42SKCfg(KuavoS42LSCfg):
    class env(KuavoS42LSCfg.env):
        gait_model_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/gait_sk120.pth'

    class rewards(KuavoS42LSCfg.rewards):
        cycle_time = 1.2
        class scales(KuavoS42LSCfg.rewards.scales):
            orientation = 1
            base_height = 0.5

            tracking_x_lin_vel = 3
            tracking_y_lin_vel = 3
            tracking_ang_vel = 3
            vel_mismatch_exp = 2.

    class init_state(KuavoS40Cfg.init_state):
        pos = [0., 0., 0.88]
        default_joint_angles = {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [0.0, 0.0, -0.27, 0.52, -0.3, 0.0]):
                default_joint_angles[f'leg_{lr}{idx}_joint'] = value
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 8), [0.15, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0]):
                default_joint_angles[f'zarm_{lr}{idx}_joint'] = value

class KuavoS42SKArmCfg(KuavoS42SKCfg):
    class init_state(KuavoS42SKCfg.init_state):
        pos = [0., 0., 0.88]
        default_joint_angles = {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [0.0, 0.0, -0.27, 0.52, -0.3, 0.0]):
                default_joint_angles[f'leg_{lr}{idx}_joint'] = value
        for idx in range(1, 8):
            default_joint_angles[f'zarm_l{idx}_joint'] = 0.
        for idx, value in zip(range(1, 8), [-10 / 57.3, -10 / 57.3, 0, -40 / 57.3, -90 / 57.3, 0, 50 / 57.3]):
            default_joint_angles[f'zarm_r{idx}_joint'] = value

class KuavoS42SK2Cfg(KuavoS42SKCfg):
    class rewards(KuavoS42SKCfg.rewards):
        class scales(KuavoS42SKCfg.rewards.scales):
            joint_pos = 10.
            joint_arm_pos = 5.
            knee_joint_pos = 10
            base_height = 0
            foot_pos = 0.
            # orientation = 1.

            tracking_x_lin_vel = 6.
            # tracking_y_lin_vel = 3.
            # tracking_ang_vel = 3.
            # vel_mismatch_exp = 2.


class KuavoS42CfgPPO(KuavoS40CfgPPO):
    class runner(KuavoS40CfgPPO.runner):
        experiment_name = 'Kuavo_s42_ppo'

    class algorithm(KuavoS40CfgPPO.algorithm):
        entropy_coef = 0.001

class KuavoS42LSCfgPPO(KuavoS42CfgPPO):
    class runner(KuavoS42CfgPPO.runner):
        policy_class_name = 'LongShortActorCritic'
        experiment_name = 'Kuavo_s42_ls_ppo'

    class algorithm(KuavoS42CfgPPO.algorithm):
        entropy_coef = 0.001
        num_mini_batches = 16

class KuavoS42SKCfgPPO(KuavoS42LSCfgPPO):
    class runner(KuavoS42LSCfgPPO.runner):
        experiment_name = 'Kuavo_s42_sk_ppo'

    class algorithm(KuavoS42LSCfgPPO.algorithm):
        entropy_coef = 0.001
