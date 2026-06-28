# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class rigid_body_masses(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        self.sum_mass = torch.sum(
            self.asset.root_physx_view.get_masses()[:, self.body_ids].to(env.device),
            dim=-1,
        ).unsqueeze(-1)
        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            self.sum_mass = torch.sum(
                self.asset.root_physx_view.get_masses()[:, self.body_ids].to(
                    env.device
                ),
                dim=-1,
            ).unsqueeze(-1)
        return self.sum_mass


class rigid_body_material(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(
            None
        ):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
        self.idxs = []
        for body_id in self.body_ids:
            idx = sum(self.num_shapes_per_body[:body_id])
            self.idxs.append(idx)

        materials = self.asset.root_physx_view.get_material_properties()
        self.materials = (
            materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
        )

        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            materials = self.asset.root_physx_view.get_material_properties()
            self.materials = (
                materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
            )
        return self.materials


class base_com(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        self.coms = (
            self.asset.root_physx_view.get_coms()[:, self.body_ids, :3]
            .to(env.device)
            .squeeze(1)
        )
        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            self.coms = (
                self.asset.root_physx_view.get_coms()[:, self.body_ids, :3]
                .to(env.device)
                .squeeze(1)
            )
        return self.coms


def contact_information(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    data = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    contact_information = torch.sum(torch.square(data), dim=-1) > 1

    return contact_information.float()


def action_delay(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuators_names: str = "base_legs",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        asset.actuators[actuators_names]
        .positions_delay_buffer.time_lags.float()
        .to(env.device)
        .unsqueeze(1)
    )


def joint_torques(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def joint_accs(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return contact_force.flatten(1, 2)


def feet_lin_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return body_lin_vel_w.flatten(1)


def push_force(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    external_force_b = asset._external_force_b[:, asset_cfg.body_ids, :]
    return external_force_b.flatten(1)


def push_torque(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    external_torque_b = asset._external_torque_b[:, asset_cfg.body_ids, :]
    return external_torque_b.flatten(1)


def feet_heights_bipeds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None,
) -> torch.Tensor:
    foot_heights = torch.stack(
        [
            env.scene[sensor_cfg.name].data.pos_w[:, 2]
            - env.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [sensor_cfg1, sensor_cfg2]
            if sensor_cfg is not None
        ],
        dim=-1,
    )
    foot_heights = torch.nan_to_num(foot_heights, nan=0, posinf=0, neginf=0)
    foot_heights = torch.clamp(foot_heights - 0.02, min=0.0)

    return foot_heights.flatten(1)


def feet_air_time_obs(
        env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    return air_time
