# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .rough_env_cfg import KuavoS42RoughEnvCfg
from ext_template.terrains import RANDOM_TERRAINS_CFG


@configclass
class KuavoS42FlatEnvCfg(KuavoS42RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
# --- 新增以下三行，暴力砍掉所有父类继承下来的网络贴图和天空！ ---
        self.scene.terrain.visual_material = None
        self.scene.sky_light = None
        self.scene.light = None

        # # override rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        # # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        #self.scene.terrain.terrain_generator = RANDOM_TERRAINS_CFG
        # no height scan
        # self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class KuavoS42FlatEnvCfg_PLAY(KuavoS42FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None

        self.events.randomize_rigid_body_com = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_link_mass = None

        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
