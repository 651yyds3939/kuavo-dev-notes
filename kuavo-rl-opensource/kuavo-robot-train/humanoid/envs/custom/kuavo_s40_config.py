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

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class KuavoS40Cfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        frame_stack_skip = 1
        c_frame_stack = 3
        num_single_obs = 47 + 3 + 14 * 3
        num_observations = int(frame_stack // frame_stack_skip * num_single_obs)
        single_num_privileged_obs = 73 + 1 + 14 * 4 + 4 + 26 * 5 + 15 + 9
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12 + 14
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False  # speed up training by using reference actions

        gait_model_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s40/gait_walk97.pth'

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s40/mjcf/biped_s40.xml'

        name = "kuavo"
        foot_name = ["leg_r6_link", "leg_l6_link"]
        knee_name = ["leg_r4_link", "leg_l4_link"]

        terminate_after_contacts_on = [
            'base_link',
            'leg_r1_link', 'leg_l1_link',
            'leg_r2_link', 'leg_l2_link',
            'leg_r3_link', 'leg_l3_link',
            'leg_r4_link', 'leg_l4_link',
            'leg_r5_link', 'leg_l5_link',
        ]
        penalize_contacts_on =[
            'base_link',
            'leg_r1_link', 'leg_l1_link',
            'leg_r2_link', 'leg_l2_link',
            'leg_r3_link', 'leg_l3_link',
            'leg_r4_link', 'leg_l4_link',
            'leg_r5_link', 'leg_l5_link',
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

        velocity_limit = [14, 14, 23, 14, 10, 10] * 2 + [10] * 14

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = True
        # mujoco use the max friction of two contact body, so here is actually the "min friction"
        static_friction = 0
        dynamic_friction = 0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 5  # number of terrain rows (levels)
        num_cols = 5  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.4, 0., 0.2, 0.2, 0, 0]
        # terrain_proportions = [0.0, 0., 0.0, .0, 1.0, 0, 0]
        restitution = 0.

        measured_points_x = [-0.3, -0.15, 0., 0.15, 0.3]
        measured_points_y = [-0.15, 0., 0.15]

    class noise:
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            lin_acc = 0.5
            last_torque = 0
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0., 0., 0.795]
        default_joint_angles = {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [0.0, 0.0, -0.44, 0.72, -0.33, 0.0]):
                default_joint_angles[f'leg_{lr}{idx}_joint'] = value
            # for idx, value in zip([1, 6], [0.02, -0.02]):
            #     default_joint_angles[f'leg_{lr}{idx}_joint'] = value * (1 if lr == 'l' else -1)
        for lr, idx in product(['l', 'r'], range(1, 8)):
            default_joint_angles[f'zarm_{lr}{idx}_joint'] = 0.

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness, damping = {}, {}

        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [60.0, 60.0, 60.0, 60.0, 15.0, 15.0]):
                stiffness[f'leg_{lr}{idx}_joint'] = value
        for lr, idx in product(['l', 'r'], range(1, 8)):
            stiffness[f'zarm_{lr}{idx}_joint'] = 5.0

        # for lr, idx in product(['l', 'r'], range(1, 7)):
        #     damping[f'leg_{lr}{idx}_joint'] = 0.5
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [34.0, 6.0, 12.0, 12.0, 22.0, 22.0]):
                damping[f'leg_{lr}{idx}_joint'] = value
        for lr, idx in product(['l', 'r'], range(1, 8)):
            damping[f'zarm_{lr}{idx}_joint'] = 3

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]

        randomize_friction = True
        friction_range = [0.2, 1.0]

        randomize_restitution = True
        restitution_range = [0., 0.5]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_joint_friction = True
        joint_friction_range = [0.5, 1.5]

        randomize_joint_armature = True
        joint_armature_range = [0.5, 1.5]

        randomize_joint_pos_bias = True
        joint_pos_bias_range = [-0.05, 0.05]

        # disturbance = True
        # disturbance_range = [-600.0, 600.0]
        # disturbance_s = 3

        push_robots = False
        push_interval_s = 6.
        push_length_s = [0.05, 0.5]
        max_push = [15, 15, 15, 5, 5, 5]

        randomize_kp = True
        kp_range = [0.8, 1.2]

        randomize_kd = True
        kd_range = [0.8, 1.2]

        # action_delay = False
        # action_delay_range = [10, 40]

        # root_states_delay = False
        # root_states_delay_range = [10, 30]
        #
        # joint_states_delay = False
        # joint_states_delay_range = [2, 15]

        randomize_euler_xy_zero_pos = False
        euler_xy_zero_pos_range = [-0.03, 0.03]

        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 5
        min_lin_vel = 0.
        min_ang_vel = 0.
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        rel_standing_envs = 0.3
        rel_marking_envs = 0.1
        rel_straight_envs = 0.4

        class ranges:
            lin_vel_x = [-0.4, 1.0]  # min max [m/s]
            lin_vel_y = [-0.2, 0.2]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]
            # lin_vel_x = [0., 0.8]  # min max [m/s]
            # lin_vel_y = [0, 0]  # min max [m/s]
            # ang_vel_yaw = [0, 0]  # min max [rad/s]
            heading = [0, 0]

    class rewards:
        joint_pos_sigma = 4.
        period_symmetric = {
            "dof_pos": {"dim_num": 26, "sigma": 4., "scale": 1.},
            # "actions": {"dim_num": 26, "sigma": 0.4, "scale": 1.},
            # "dof_vel": {"dim_num": 26, "sigma": 0.4, "scale": 0.25},
            # "torques": {"dim_num": 26, "sigma": 0.04, "scale": 0.25},
        }
        base_height_target = 0.795
        foot_height = 0.045
        min_dist = 0.25
        max_dist = 0.6
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.24  # rad
        target_feet_height = 0.12  # m
        cycle_time = 0.97  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        x_tracking_sigmas = [6, 60]
        y_tracking_sigmas = [6, 60]
        yaw_tracking_sigmas = [6, 60]
        max_contact_force = 550  # Forces above this value are penalized

        class scales:
            # task
            joint_pos = 10.
            half_period = 2.
            foot_slip = -2.  # original -0.05
            foot_pos = 5.
            # feet_distance = 0.2
            # knee_distance = 0.2
            feet_contact_forces = -0.05  # original -0.01
            tracking_x_lin_vel = 1.5
            tracking_y_lin_vel = 1.5
            tracking_ang_vel = 3
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            # track_vel_hard = 0.5
            orientation = 2.
            base_height = 1.
            base_acc = 0.5  # original 0.2
            # marking_time = 1

            feet_contact_same = 1
            feet_contact_number = 20

            # reg
            action_smoothness = -0.01  # original -0.002
            torques = -5e-5  # original -1e-5
            dof_vel = -5e-3  # original -5e-4
            dof_acc = -1e-7  # original -1e-7
            # dof_jerk = -1e-8

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            lin_acc = 0.5
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        clip_observations = 18.
        clip_actions = 18.


class KuavoS40LSCfg(KuavoS40Cfg):
    class env(KuavoS40Cfg.env):
        frame_stack = 100
        frame_stack_skip = 1
        num_single_obs = 47 + 3 + 14 * 3
        num_observations = int(frame_stack // frame_stack_skip * num_single_obs)


class KuavoS40C0Cfg(KuavoS40Cfg):
    class domain_rand(KuavoS40Cfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_com_displacement = False
        push_robots = False
    class rewards(KuavoS40Cfg.rewards):
        class scales:
            joint_pos = 5.
            half_period = 1.
            tracking_x_lin_vel = 1.
            tracking_y_lin_vel = 1.
            tracking_ang_vel = 1.

class KuavoS40SKCfg(KuavoS40LSCfg):
    pass
    # class rewards(KuavoS40LSCfg.rewards):
    #     class scales(KuavoS40LSCfg.rewards.scales):
    #         joint_pos = 10.
    #         foot_pos = 2.


class KuavoS40TrotCfg(KuavoS40Cfg):

    class env(KuavoS40Cfg.env):
        gait_model_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s40/gait_trot.pth'

    class rewards(KuavoS40Cfg.rewards):
        cycle_time = 0.75

    class commands(KuavoS40Cfg.commands):
        min_lin_vel = 0.
        min_ang_vel = 0.
        rel_standing_envs = 0.

        class ranges(KuavoS40Cfg.commands.ranges):
            lin_vel_x = [0., 0.]  # min max [m/s]
            lin_vel_y = [0., 0.]  # min max [m/s]
            ang_vel_yaw = [0., 0.]  # min max [rad/s]
            heading = [0, 0]

class KuavoS40AmassCfg(KuavoS40Cfg):
    class env(KuavoS40Cfg.env):
        gait_model_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s40/gait_amass.pth'

    class init_state(KuavoS40Cfg.init_state):
        pos = [0., 0., 0.82]
        default_joint_angles = {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [0.0, 0.0, -0.29, 0.54, -0.25, 0.0]):
                default_joint_angles[f'leg_{lr}{idx}_joint'] = value
        for lr, idx in product(['l', 'r'], range(1, 8)):
            default_joint_angles[f'zarm_{lr}{idx}_joint'] = 0.

    # class rewards(KuavoS40Cfg.rewards):
    #     joint_pos_sigma = 3.
    #
    #     # class scales(KuavoS40Cfg.rewards.scales):
    #     #     joint_pos = 3.
    #     #     orientation = 0.5
    #     #
    #     #     feet_contact_forces = -0.1

    class commands(KuavoS40Cfg.commands):
        min_lin_vel = 0.6
        min_ang_vel = 0.

        rel_standing_envs = 0.

        class ranges:
            lin_vel_x = [0.0, 1.6]  # min max [m/s]
            lin_vel_y = [0, 0]  # min max [m/s]
            ang_vel_yaw = [0, 0]  # min max [rad/s]
            heading = [0, 0]

    class domain_rand(KuavoS40Cfg.domain_rand):
        push_robots = False
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4


class KuavoS40CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

        use_clipped_value_loss = False

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 100000  # number of policy updates

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'Kuavo_s40_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

class KuavoS40LSCfgPPO(KuavoS40CfgPPO):
    class runner(KuavoS40CfgPPO.runner):
        policy_class_name = 'LongShortActorCritic'
        experiment_name = 'Kuavo_s40_ls_ppo'

    class algorithm(KuavoS40CfgPPO.algorithm):
        entropy_coef = 0.001
        num_mini_batches = 16

class KuavoS40SKCfgPPO(KuavoS40LSCfgPPO):
    class runner(KuavoS40LSCfgPPO.runner):
        experiment_name = 'Kuavo_s40_sk_ppo'

    class algorithm(KuavoS40LSCfgPPO.algorithm):
        entropy_coef = 0.001

class KuavoS40TrotCfgPPO(KuavoS40CfgPPO):
    class runner(KuavoS40CfgPPO.runner):
        experiment_name = 'Kuavo_s40_trot_ppo'

    class algorithm(KuavoS40CfgPPO.algorithm):
        entropy_coef = 0.001

class KuavoS40AmassCfgPPO(KuavoS40CfgPPO):
    class runner(KuavoS40CfgPPO.runner):
        experiment_name = 'Kuavo_s40_amass_ppo'

    class algorithm(KuavoS40CfgPPO.algorithm):
        entropy_coef = 0.0003


