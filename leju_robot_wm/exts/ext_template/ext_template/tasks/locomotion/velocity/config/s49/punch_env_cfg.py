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
from ext_template.assets.kuavo import Kuavos49_CFG

# 全身原地舞：左右腿交替律动 + 手臂完整编舞（adapt_dance_csv.py --profile fullbody_inplace）
DANCE_CSV = "kuavo_action_S49_FROM_S54_INPLACE_RAD.csv"

# RL 仅控制 26 关节（与 S46 USD / 部署 policy / CSV 顺序一致）
KUAVO_RL_JOINT_NAMES = [
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
    "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
]


def _rl_joint_scene_cfg() -> SceneEntityCfg:
    """Fresh SceneEntityCfg per obs term (reuse causes joint_names/joint_ids conflict after resolve)."""
    return SceneEntityCfg("robot", joint_names=KUAVO_RL_JOINT_NAMES, preserve_order=True)


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
        # 当前帧参考关节角（相对默认姿态），让策略直接看到手脚目标，而不只靠相位猜
        reference_joint_pos = ObsTerm(
            func=local_rewards.reference_joint_pos_rel,
            params={"csv_path": DANCE_CSV},
        )
        actions = ObsTerm(func=mdp.last_action)
        
        # 🟢 动态相位观测：读取CSV动态计算长度，确保观测与目标绝对同步
        action_phase = ObsTerm(func=local_rewards.action_phase, params={"csv_path": DANCE_CSV})

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
    """全身舞跟踪 + 原地稳定约束（减少前栽/滑步）"""
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
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    base_lin_vel_xy_stationary = RewTerm(
        func=local_rewards.base_lin_vel_xy_l2_stationary,
        weight=-1.3,
    )
    base_ang_vel_yaw_stationary = RewTerm(
        func=local_rewards.base_ang_vel_yaw_l2_stationary,
        weight=-0.3,
    )
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
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.004)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.006)

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
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "threshold_min": 0.2,
            "threshold_max": 0.5
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.12,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r][1,2]_joint"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)
    base_height = RewTerm(
        func=local_rewards.base_height_l2,
        weight=-4.0,
        params={"target_height": 0.85},
    )
    contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "threshold": 900,
            "violation_max": 300,
        },
    )
    # 速度指令恒为 0 时该惩罚会压制所有关节运动，与模仿奖励冲突，必须关闭
    stand_still_without_cmd = RewTerm(func=mdp.stand_still_without_cmd, weight=0.0, params={"command_name": "base_velocity"})
    gravity_aligned_when_stopping = RewTerm(func=mdp.gravity_aligned_when_stopping, weight=0.0, params={"command_name": "base_velocity"})

    track_punch_arms = RewTerm(
        func=local_rewards.track_punch_arms_trajectory_upright_exp,
        weight=15.0,
        params={"std": 0.38, "csv_path": DANCE_CSV, "min_upright": 0.75},
    )
    track_punch_legs = RewTerm(
        func=local_rewards.track_punch_legs_trajectory_upright_exp,
        weight=9.0,
        params={"std": 0.50, "csv_path": DANCE_CSV, "min_upright": 0.75},
    )
    arm_roll_penalty = RewTerm(
        func=local_rewards.penalty_arm_roll_limit,
        weight=-3.0,
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
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.3, 0.3)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.95, 1.05), "velocity_range": (0.0, 0.0)},
    )
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
class KuavoS49PunchEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = Kuavos49_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.30
        self.actions.joint_pos.joint_names = KUAVO_RL_JOINT_NAMES

        self.observations.policy.joint_pos.params = {"asset_cfg": _rl_joint_scene_cfg()}
        self.observations.policy.joint_vel.params = {"asset_cfg": _rl_joint_scene_cfg()}
        self.observations.critic.joint_pos.params = {"asset_cfg": _rl_joint_scene_cfg()}
        self.observations.critic.joint_vel.params = {"asset_cfg": _rl_joint_scene_cfg()}
        self.observations.critic.joint_torques.params = {"asset_cfg": _rl_joint_scene_cfg()}
        self.observations.critic.joint_accs.params = {"asset_cfg": _rl_joint_scene_cfg()}

        # S49 全 URDF 碰撞体较多，8192 env 易触发 patch buffer overflow
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**27


@configclass
class KuavoS49PunchEnvCfg_PLAY(KuavoS49PunchEnvCfg):
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