# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

# 1. 导入官方的基础配置
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

# 2. 【核心修复】：恢复使用本地mdp，以确保能找到全部基础和自定义的观测/奖励项
import ext_template.tasks.locomotion.velocity.mdp as mdp
import ext_template.tasks.locomotion.velocity.mdp.rewards as local_rewards

# 3. 统一导入管理器依赖
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from ext_template.terrains import ROUGH_TERRAINS_CFG
from ext_template.assets.kuavo import Kuavos46_CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # 保持纯平模式
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.4,
            dynamic_friction=0.4,
            restitution=0.5,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    
    # 关闭物理雷达，防止干扰
    height_scanner = None
    Feet_L_scanner = None
    Feet_R_scanner = None
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # 🟢 全身模仿模式下，无需生成任何随机速度指令，防止干扰观测网络
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        
        # 🟢 动态相位观测：读取CSV动态计算长度，确保观测与目标绝对同步
        action_phase = ObsTerm(func=local_rewards.action_phase, params={"csv_path": "kuavo_action_PERFECT_LIMIT_RAD.csv"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_torques = ObsTerm(func=mdp.joint_torques)
        joint_accs = ObsTerm(func=mdp.joint_accs)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])},
        )
        feet_contact_force = ObsTerm(
            func=mdp.feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_[l,r]6_link"])},
        )
        base_mass_rel = ObsTerm(
            func=mdp.rigid_body_masses,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        rigid_body_material = ObsTerm(
            func=mdp.rigid_body_material,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])},
        )
        base_com = ObsTerm(
            func=mdp.base_com,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        action_delay = ObsTerm(func=mdp.action_delay, params={"actuators_names": "motor"})
        push_force = ObsTerm(
            func=mdp.push_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        push_torque = ObsTerm(
            func=mdp.push_torque,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        feet_air_times = ObsTerm(
            func=mdp.feet_air_time_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """满血版平衡机理 + 动作跟踪体系"""
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_power_l2 = RewTerm(func=mdp.joint_power_l2, weight=-2.0e-5)
    
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r][1-5]_joint", "zarm_.*_joint"])},
    )
    dof_torques_ankle_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]6_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_[l,r][1-5]_link", "base_link", "zarm_.*_link"]),
            "threshold": 1.0,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_clip,
        weight=0.0, # 🔴 禁止强迫抬腿，完全交由CSV轨迹控制脚部
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "threshold_min": 0.2,
            "threshold_max": 0.5
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0, # 🔴 关闭防滑步，释放武术转体滑步的自由度
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r][1,2]_joint"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.25)
    contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "threshold": 900,
            "violation_max": 300,
        },
    )
    stand_still_without_cmd = RewTerm(func=mdp.stand_still_without_cmd, weight=0.0, params={"command_name": "base_velocity"})
    gravity_aligned_when_stopping = RewTerm(func=mdp.gravity_aligned_when_stopping, weight=0.0, params={"command_name": "base_velocity"}) # 🔴 防止强迫直立腰杆，释放弯腰自由度
    
    # 🥊 核心：指向 local_rewards 的黄金打拳跟随奖励（补齐显式 CSV 路径参数）
    track_punch_imitation = RewTerm(
        func=local_rewards.track_punch_joint_trajectory_exp,
        weight=15.0,
        params={"std": 0.5, "csv_path": "kuavo_action_PERFECT_LIMIT_RAD.csv"}, 
    )
    # 🛡️ 硬件级防穿模限制惩罚
    arm_roll_penalty = RewTerm(
        func=local_rewards.penalty_arm_roll_limit,
        weight=-5.0,
    )


@configclass
class TerminationsCfg:
    """核心修复：将躯干重置连杆精准纠正回 base_link"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 50.0},
    )
    dof_pos_illegal = DoneTerm(func=mdp.dof_pos_illegal, params={"actuators_names": "motor"})


@configclass
class EventCfg:
    """Domain Randomization 域随机化：激活全套稳定干扰机制"""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "mass_distribution_params": (-5.0, 5.0), "operation": "add"},
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_.*_link", "zarm_.*_link"]), "mass_distribution_params": (0.8, 1.2), "operation": "scale"},
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_base_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}},
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.8, 1.2), "operation": "scale"},
    )
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "friction_distribution_params": (1.0, 1.0), "armature_distribution_params": (0.5, 1.5), "operation": "scale"},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3)},
        },
    )
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_scale, mode="reset", params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)})
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": {"x": (-2500.0, 2500.0), "y": (-2500.0, 2500.0), "z": (-1500.0, 1500.0)},
            "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
            "probability": 0.002,
        },
    )


@configclass
class CurriculumCfg:
    terrain_levels = None


@configclass
class KuavoS42PunchEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5) 
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = Kuavos46_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.25


@configclass
class KuavoS42PunchEnvCfg_PLAY(KuavoS42PunchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.events.base_external_force_torque = None
        
        # 显存防爆设置
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**27