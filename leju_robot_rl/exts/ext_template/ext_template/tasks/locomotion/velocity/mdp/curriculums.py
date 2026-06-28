"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers import RewardManager


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    reward: RewardManager = env.reward_manager
    lin_track_reward_sum = (
        reward._episode_sums["track_lin_vel_xy_exp"][env_ids] / env.max_episode_length_s
    )
    lin_track_reward_idx = reward._term_names.index("track_lin_vel_xy_exp")
    lin_track_reward_weight = reward._term_cfgs[lin_track_reward_idx].weight
    ang_track_reward_sum = (
        reward._episode_sums["track_ang_vel_z_exp"][env_ids] / env.max_episode_length_s
    )
    ang_track_reward_idx = reward._term_names.index("track_ang_vel_z_exp")
    ang_track_reward_weight = reward._term_cfgs[ang_track_reward_idx].weight
    # compute the distance the robot walked
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    # robots that walked far enough progress to harder terrains
    move_up = (
        (distance > terrain.cfg.terrain_generator.size[0] / 2)
        & (lin_track_reward_sum > lin_track_reward_weight * 0.7)
        & (ang_track_reward_sum > ang_track_reward_weight * 0.7)
    )
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (lin_track_reward_sum < lin_track_reward_weight * 0.6) | (
        ang_track_reward_sum < ang_track_reward_weight * 0.6
    )
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
