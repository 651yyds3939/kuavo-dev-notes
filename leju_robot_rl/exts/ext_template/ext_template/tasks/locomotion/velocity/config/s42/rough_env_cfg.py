# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
# from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip
from ext_template.assets.kuavo import Kuavos46_CFG

import ext_template.tasks.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
import math
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.sensors import RayCasterCfg, patterns


from dataclasses import MISSING
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ext_template.terrains import ROUGH_TERRAINS_CFG


@configclass
class MySceneCfg(InteractiveSceneCfg): #搭建拍摄片场
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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
        # visual_material=sim_utils.MdlFileCfg(
        #     mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        #     project_uvw=True,
        #     texture_scale=(0.25, 0.25),
        # ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=750.0,
    #         #texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )
    Feet_L_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_l6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    Feet_R_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_r6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class CommandsCfg: #导演发号施令
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ObservationsCfg: #机器人的感官网络 (⚠️ 核心概念)
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup): #演员只能看到基础信息
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg): #裁判拥有上帝视角
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        joint_torques = ObsTerm(func=mdp.joint_torques)
        joint_accs = ObsTerm(func=mdp.joint_accs)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])
            },
        )
        feet_contact_force = ObsTerm(
            func=mdp.feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["leg_[l,r]6_link"]
                )
            },
        )
        base_mass_rel = ObsTerm(
            func=mdp.rigid_body_masses,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        rigid_body_material = ObsTerm(
            func=mdp.rigid_body_material,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])
            },
        )
        base_com = ObsTerm(
            func=mdp.base_com,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        action_delay = ObsTerm(
            func=mdp.action_delay, params={"actuators_names": "motor"}
        )
        push_force = ObsTerm(
            func=mdp.push_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        push_torque = ObsTerm(
            func=mdp.push_torque,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )

        feet_heights = ObsTerm(
            func=mdp.feet_heights_bipeds,
            params={
                "sensor_cfg1": SceneEntityCfg("Feet_L_scanner"),
                "sensor_cfg2": SceneEntityCfg("Feet_R_scanner"),
            },
        )
        feet_air_times = ObsTerm(
            func=mdp.feet_air_time_obs,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names="leg_[l,r]6_link"
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
 
@configclass
class RewardsCfg:             #重点，决定了“考机器人哪些科目”以及“每门课占多少分”
    """Reward terms for the MDP."""

    # # # -- task
    track_lin_vel_xy_exp = RewTerm(       #线速度跟踪奖励
        func=mdp.track_lin_vel_xy_yaw_frame_exp,  #指数核函数，可以放大小误差的影响，促使机器人更快地学会跟踪命令
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_power_l2 = RewTerm(func=mdp.joint_power_l2, weight=-2.0e-5)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["leg_[l,r][1-5]_joint", "zarm_.*_joint"]
            )
        },
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
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["leg_[l,r][1-5]_link", "base_link", "zarm_.*_link"],
            ),
            "threshold": 1.0,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_clip,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "threshold_min": 0.2,
            "threshold_max": 0.5
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r][1,2]_joint"])
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "zarm_.*",
                ],
            )
        },
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.25,
    )

    contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "threshold": 900,
            "violation_max": 300,
        },
    )

    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=0.0,
        params={
            "command_name": "base_velocity",
        },
    )

    gravity_aligned_when_stopping = RewTerm(
        func=mdp.gravity_aligned_when_stopping,
        weight=0.1,
        params={
            "command_name": "base_velocity",
        },
    )
    # =========================================================================
    # 🔥 黄金魔改：在此类尾部，强行挂载我们在 rewards.py 里追加的追踪函数
    # =========================================================================
    track_punch_imitation = RewTerm(
        func=mdp.track_punch_joint_trajectory_exp,
        weight=15.0,  # 给予绝对统治级的高权重
        params={"std": 0.25}, 
    )
    


@configclass
class TerminationsCfg: #终止条件，决定了“考试不及格的条件”
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )

    dof_pos_illegal = DoneTerm(
        func=mdp.dof_pos_illegal,
        params={
            "actuators_names": "motor"
        },
    )


@configclass
class EventCfg: #九九八十一难 (Sim2Real 域随机化)
    """Configuration for events."""

    # startup
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
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_.*_link", "zarm_.*_link"]
            ),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_base_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": {
                "x": (-2500.0, 2500.0),
                "y": (-2500.0, 2500.0),
                "z": (-1500.0, 1500.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
            "probability": 0.002,  # Expect step = 1 / probability
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class KuavoS42PunchEnvCfg(LocomotionVelocityRoughEnvCfg): # 🔴 修改类名
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5) 
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = Kuavos46_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # reduce action scale
        self.actions.joint_pos.scale = 0.25


@configclass
class KuavoS42PunchEnvCfg_PLAY(KuavoS42PunchEnvCfg): # 更改本类名，并精准继承打拳基类
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # remove random pushing event
        self.events.base_external_force_torque = None
