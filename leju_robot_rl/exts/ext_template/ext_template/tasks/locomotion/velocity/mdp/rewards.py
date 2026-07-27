
from __future__ import annotations

import torch
import math
import os
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.utils.math import quat_apply, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedEnv

# 全局缓存动作特征，防止每次步进都重复读取CSV影响帧率
_PUNCH_TRAJECTORY_CACHE = {}
_GYM_JOINT_IDS_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
NUM_JOINTS = 26

# CSV / gym / deploy joint order (leg12 + arm14)
KUAVO_GYM_JOINT_NAMES = [
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
    "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
]
LEG_GYM_JOINT_NAMES = KUAVO_GYM_JOINT_NAMES[:12]
ARM_GYM_JOINT_NAMES = KUAVO_GYM_JOINT_NAMES[12:]


def _resolve_gym_joint_ids(asset: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """Map gym/CSV joint order to Isaac articulation indices (cached per asset instance)."""
    key = id(asset)
    if key not in _GYM_JOINT_IDS_CACHE:
        ids, _ = asset.find_joints(KUAVO_GYM_JOINT_NAMES, preserve_order=True)
        _GYM_JOINT_IDS_CACHE[key] = (torch.tensor(ids, device=asset.device, dtype=torch.long), _)
    return _GYM_JOINT_IDS_CACHE[key]


def _resolve_named_joint_ids(asset: Articulation, joint_names: list[str]) -> torch.Tensor:
    ids, _ = asset.find_joints(joint_names, preserve_order=True)
    return torch.tensor(ids, device=asset.device, dtype=torch.long)


def _upright_factor(asset: Articulation, min_upright: float = 0.75) -> torch.Tensor:
    """1.0 when torso is upright; 0.0 when fallen (uses body-frame projected gravity)."""
    z = -asset.data.projected_gravity_b[:, 2]
    return torch.clamp((z - min_upright) / max(1.0 - min_upright, 1e-6), 0.0, 1.0)


def _height_factor(asset: Articulation, min_height: float, target_height: float) -> torch.Tensor:
    """1.0 near target standing height; 0.0 when root drops below min_height (blocks kneel-to-mimic cheat)."""
    h = asset.data.root_link_pos_w[:, 2]
    return torch.clamp((h - min_height) / max(target_height - min_height, 1e-6), 0.0, 1.0)


def _standing_gate(
    asset: Articulation,
    min_upright: float,
    min_height: float | None,
    target_height: float,
) -> torch.Tensor:
    gate = _upright_factor(asset, min_upright)
    if min_height is not None:
        gate = gate * _height_factor(asset, min_height, target_height)
    return gate


def penalty_root_height_below(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty when pelvis/root drops below min_height (discourages kneel/crouch cheat)."""
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    return torch.square(torch.clamp(min_height - h, min=0.0))


def _parse_csv_joint_rows(full_csv_path: str) -> list[list[float]]:
    """Parse dance CSV into exactly 26 joint columns per frame."""
    rows: list[list[float]] = []
    with open(full_csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",") if x]
            if len(vals) < NUM_JOINTS:
                continue
            # time + 26 joints -> drop time column
            if len(vals) >= NUM_JOINTS + 1:
                vals = vals[1 : NUM_JOINTS + 1]
            else:
                vals = vals[:NUM_JOINTS]
            rows.append(vals)
    if not rows:
        raise ValueError(f"No valid 26-DoF rows found in CSV: {full_csv_path}")
    return rows


def _load_csv_to_cache(csv_path: str, device: torch.device):
    """辅助函数：将CSV加载进GPU显存，并自动对齐为 26 维关节轨迹"""
    if csv_path not in _PUNCH_TRAJECTORY_CACHE:
        full_csv_path = csv_path
        if not os.path.isabs(full_csv_path):
            full_csv_path = os.path.abspath(full_csv_path)
        traj = _parse_csv_joint_rows(full_csv_path)
        _PUNCH_TRAJECTORY_CACHE[csv_path] = torch.tensor(traj, dtype=torch.float32, device=device)
    return _PUNCH_TRAJECTORY_CACHE[csv_path]

def action_phase(env: ManagerBasedRLEnv, csv_path: str) -> torch.Tensor:
    """
    动态计算当前步数在 CSV 轨迹中的真实相位。
    """
    traj_tensor = _load_csv_to_cache(csv_path, env.device)
    num_frames = traj_tensor.shape[0]

    if not hasattr(env, "episode_length_buf"):
        frame_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        frame_indices = (env.episode_length_buf % num_frames).long()
    
    phase = frame_indices.float() / num_frames
    
    return torch.cat([
        torch.sin(2 * math.pi * phase).unsqueeze(1),
        torch.cos(2 * math.pi * phase).unsqueeze(1)
    ], dim=-1)


def reference_joint_pos_rel(
    env: ManagerBasedRLEnv,
    csv_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current-frame CSV reference joint positions relative to default standing pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_joint_pos = _get_csv_target_joint_pos(env, csv_path)
    joint_ids, _ = _resolve_gym_joint_ids(asset)
    default_joint_pos = asset.data.default_joint_pos[:, joint_ids]
    return target_joint_pos - default_joint_pos


def reference_joint_pos_rel_arms_csv_legs_standing(
    env: ManagerBasedRLEnv,
    csv_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference observation for arms-only dance: legs use fixed standing pose, arms use CSV.

    v16 still exposed moving CSV leg targets in the observation while rewarding fixed legs,
    which gives the policy conflicting signals and can cause leg trembling. This keeps the
    26-D observation shape unchanged but aligns the reference with the reward design.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_joint_pos = _get_csv_target_joint_pos(env, csv_path).clone()
    standing = torch.tensor(_KUAVO_STANDING_LEG_POS, device=env.device, dtype=torch.float32)
    target_joint_pos[:, :12] = standing.unsqueeze(0)
    joint_ids, _ = _resolve_gym_joint_ids(asset)
    default_joint_pos = asset.data.default_joint_pos[:, joint_ids]
    return target_joint_pos - default_joint_pos


def _get_csv_target_joint_pos(env: ManagerBasedRLEnv, csv_path: str) -> torch.Tensor:
    traj_tensor = _load_csv_to_cache(csv_path, env.device)
    num_frames = traj_tensor.shape[0]
    if not hasattr(env, "episode_length_buf"):
        frame_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        frame_indices = (env.episode_length_buf % num_frames).long()
    return traj_tensor[frame_indices]


def _track_joint_names_exp(
    env: ManagerBasedRLEnv,
    std: float,
    csv_path: str,
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_named_joint_ids(asset, joint_names)
    name_to_gym_idx = {name: i for i, name in enumerate(KUAVO_GYM_JOINT_NAMES)}
    gym_indices = [name_to_gym_idx[name] for name in joint_names]
    target_joint_pos = _get_csv_target_joint_pos(env, csv_path)[:, gym_indices]
    current_joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_pos_error = torch.mean(torch.square(current_joint_pos - target_joint_pos), dim=1)
    return torch.exp(-joint_pos_error / std**2)


def _track_joint_slice_exp(
    env: ManagerBasedRLEnv,
    std: float,
    csv_path: str,
    joint_start: int,
    joint_end: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    joint_names = KUAVO_GYM_JOINT_NAMES[joint_start:joint_end]
    return _track_joint_names_exp(env, std, csv_path, joint_names, asset_cfg)


def track_punch_joint_trajectory_exp(
    env: ManagerBasedRLEnv,
    std: float,
    csv_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """全身 26 关节 CSV 轨迹指数跟踪奖励。"""
    return _track_joint_slice_exp(env, std, csv_path, 0, NUM_JOINTS, asset_cfg)


def track_punch_legs_trajectory_exp(
    env: ManagerBasedRLEnv,
    std: float,
    csv_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """下肢 12 关节 CSV 轨迹跟踪。"""
    return _track_joint_slice_exp(env, std, csv_path, 0, 12, asset_cfg)


def track_punch_arms_trajectory_exp(
    env: ManagerBasedRLEnv,
    std: float,
    csv_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """上肢 14 关节 CSV 轨迹跟踪。"""
    return _track_joint_names_exp(env, std, csv_path, ARM_GYM_JOINT_NAMES, asset_cfg)


def track_punch_arms_trajectory_upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    csv_path: str,
    min_upright: float = 0.75,
    min_height: float | None = None,
    target_height: float = 0.87,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    track = _track_joint_names_exp(env, std, csv_path, ARM_GYM_JOINT_NAMES, asset_cfg)
    return track * _standing_gate(asset, min_upright, min_height, target_height)


def track_punch_legs_trajectory_upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    csv_path: str,
    min_upright: float = 0.75,
    min_height: float | None = None,
    target_height: float = 0.87,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    track = _track_joint_names_exp(env, std, csv_path, LEG_GYM_JOINT_NAMES, asset_cfg)
    return track * _standing_gate(asset, min_upright, min_height, target_height)


def penalty_foot_pitch_deviation(
    env: ManagerBasedRLEnv,
    csv_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize ankle pitch (leg_l5/r5) deviating from CSV reference to discourage toe-up cheat."""
    asset: Articulation = env.scene[asset_cfg.name]
    ankle_names = ["leg_l5_joint", "leg_r5_joint"]
    joint_ids = _resolve_named_joint_ids(asset, ankle_names)
    name_to_gym_idx = {name: i for i, name in enumerate(KUAVO_GYM_JOINT_NAMES)}
    gym_indices = [name_to_gym_idx[name] for name in ankle_names]
    target_joint_pos = _get_csv_target_joint_pos(env, csv_path)[:, gym_indices]
    current_joint_pos = asset.data.joint_pos[:, joint_ids]
    return torch.mean(torch.square(current_joint_pos - target_joint_pos), dim=1)


def penalty_foot_link_flat_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
    min_upright: float = 0.75,
) -> torch.Tensor:
    """Penalize foot link tilt while upright and in contact (toe-up / heel-only cheat).

    Uses foot-link +Z alignment with world up (robust vs URDF frame conventions).
    Gated by torso upright so fallen robots do not receive spurious gradients.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :].norm(dim=-1)
        > contact_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    num_feet = foot_quat_w.shape[1]
    local_up = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
    local_up = local_up.view(1, 1, 3).expand(env.num_envs, num_feet, 3)
    quat_flat = foot_quat_w.reshape(-1, 4)
    up_flat = local_up.reshape(-1, 3)
    foot_up_w = quat_apply(quat_flat, up_flat).view(env.num_envs, num_feet, 3)
    # flat sole: foot +Z ~ world +Z  →  foot_up_w[:,:,2] ~ 1
    misalign = torch.square(1.0 - torch.clamp(foot_up_w[:, :, 2], min=-1.0, max=1.0))
    gate = _upright_factor(asset, min_upright).unsqueeze(1)
    return torch.sum(misalign * contacts * gate, dim=1)


def feet_contact_force_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize high foot contact force × velocity (discourage unstable heel scraping)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_norm = torch.norm(
        contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :], dim=-1
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    velocity_norm = torch.norm(body_vel, dim=-1)
    violation = torch.clamp(
        contact_force_norm * velocity_norm - force_velocity_threshold, min=0.0
    )
    return torch.sum(violation / force_velocity_threshold, dim=1)


def base_lin_vel_xy_l2_stationary(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for root XY linear velocity to discourage forward stumbling during dance."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_b = quat_rotate_inverse(asset.data.root_link_quat_w, asset.data.root_com_lin_vel_w[:, :3])
    return torch.sum(torch.square(vel_b[:, :2]), dim=1)


def penalty_root_xy_displacement(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for drifting away from episode spawn in XY (in-place dance anchor)."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_local = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.sum(torch.square(pos_local), dim=1)


def penalty_leg_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize rapid leg joint oscillation (high-frequency jitter while tracking CSV)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_named_joint_ids(asset, LEG_GYM_JOINT_NAMES)
    return torch.sum(torch.square(asset.data.joint_vel[:, joint_ids]), dim=1)


def penalty_leg_joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize leg joint acceleration spikes (smooth out trembling)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_named_joint_ids(asset, LEG_GYM_JOINT_NAMES)
    return torch.sum(torch.square(asset.data.joint_acc[:, joint_ids]), dim=1)


def penalty_feet_motion_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot-link motion in all directions for an arms-only in-place policy."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(foot_vel), dim=(1, 2))


def penalty_feet_airborne(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    grace_time: float = 0.04,
) -> torch.Tensor:
    """Penalize sustained foot lifting while allowing brief contact-estimation flicker."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(air_time - grace_time, min=0.0), dim=1)


# Kuavo S49 默认站姿（来自 CSV 第1帧，膝盖弯曲 ~30° 踝关节 ~-17°）
_KUAVO_STANDING_LEG_POS = [
    -0.0000, -0.0002, -0.2701, 0.5199, -0.3000, -0.0000,
    -0.0000, -0.0002, -0.2697, 0.5200, -0.3003, -0.0000,
]


def penalty_leg_deviation_standing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize leg joints deviating from the natural bent-knee standing posture."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_named_joint_ids(asset, LEG_GYM_JOINT_NAMES)
    standing = torch.tensor(_KUAVO_STANDING_LEG_POS, device=env.device, dtype=torch.float32)
    current = asset.data.joint_pos[:, joint_ids]
    return torch.sum(torch.abs(current - standing), dim=1)


def track_leg_standing_upright_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.30,
    min_upright: float = 0.85,
    min_height: float | None = 0.72,
    target_height: float = 0.87,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward legs for maintaining natural standing posture (knee ~30°, ankle ~-17°).
    
    Critical insight from v2/v15 comparison: This is NOT just a constraint, it's the
    stability pillar. The exp+gate structure creates a bounded [0,1] attractor that
    PPO loves, identical to v2's track_punch_legs but targeting fixed standing pose
    instead of CSV dancing legs.
    
    This simultaneously:
    1. Inherits v2's stability (same reward structure)
    2. Keeps legs still (target = standing pose, not CSV)
    3. Prevents hook-foot (ankle target = -0.30 rad = natural flat)
    
    Why exp+gate beats L1 penalty (v15's joint_deviation_legs):
    - Positive feedback ("stand well = get paid") vs negative ("move = lose money")
    - Bounded [0,1] vs unbounded negative
    - Upright-gated (syncs with arm rewards) vs always-on
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_named_joint_ids(asset, LEG_GYM_JOINT_NAMES)
    standing = torch.tensor(_KUAVO_STANDING_LEG_POS, device=env.device, dtype=torch.float32)
    current = asset.data.joint_pos[:, joint_ids]
    err = torch.mean(torch.square(current - standing), dim=1)
    track_reward = torch.exp(-err / std**2)
    return track_reward * _standing_gate(asset, min_upright, min_height, target_height)


def track_ankle_pitch_standing_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.12,
    min_upright: float = 0.85,
    min_height: float | None = 0.72,
    target_height: float = 0.87,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dedicated positive reward for keeping ankle pitch at the natural flat-foot angle.

    The whole-leg standing reward averages 12 joints, so ankle hook-foot can be diluted.
    This term isolates leg_l5/leg_r5 and softly attracts them to -0.30 rad without using
    foot-link flatness constraints, which previously caused jitter or early death.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ankle_names = ["leg_l5_joint", "leg_r5_joint"]
    joint_ids = _resolve_named_joint_ids(asset, ankle_names)
    target = torch.tensor([-0.3000, -0.3003], device=env.device, dtype=torch.float32)
    current = asset.data.joint_pos[:, joint_ids]
    err = torch.mean(torch.square(current - target), dim=1)
    return torch.exp(-err / std**2) * _standing_gate(asset, min_upright, min_height, target_height)


def penalty_ankle_pitch_standing_soft(
    env: ManagerBasedRLEnv,
    tolerance: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Soft dead-zone penalty for ankle pitch deviation from flat-foot standing angle.

    Uses a tolerance so the policy can still make small balance corrections. This is much
    gentler than the failed foot-link flat constraints and should not dominate stability.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ankle_names = ["leg_l5_joint", "leg_r5_joint"]
    joint_ids = _resolve_named_joint_ids(asset, ankle_names)
    target = torch.tensor([-0.3000, -0.3003], device=env.device, dtype=torch.float32)
    current = asset.data.joint_pos[:, joint_ids]
    violation = torch.clamp(torch.abs(current - target) - tolerance, min=0.0)
    return torch.mean(torch.square(violation), dim=1)


def _reset_all_joints(
    env: ManagerBasedEnv, env_ids: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor
):
    """Set joints into sim (helper to avoid code duplication)."""
    asset: Articulation = env.scene["robot"]
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_legs_to_standing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Spawn leg joints at natural standing posture (knees ~30°) with ±5% noise;
    arms reset to default with ±5% noise (same as standard reset_joints_by_scale)."""
    asset: Articulation = env.scene[asset_cfg.name]

    leg_ids = _resolve_named_joint_ids(asset, LEG_GYM_JOINT_NAMES)
    standing = torch.tensor(_KUAVO_STANDING_LEG_POS, device=env.device, dtype=torch.float32)

    n = len(env_ids)
    n_all = asset.data.default_joint_pos.shape[1]

    # start from default (URDF zero → arm reference)
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros(n, n_all, device=env.device)

    # arms: scale default by ±5% (same as standard reset_joints_by_scale)
    arm_noise = (0.95 + 0.10 * torch.rand(n, n_all, device=env.device))
    joint_pos *= arm_noise

    # legs: OVERWRITE with standing posture + ±5% noise
    leg_noise = 0.95 + 0.10 * torch.rand(n, len(standing), device=env.device)
    joint_pos[:, leg_ids] = standing.unsqueeze(0) * leg_noise

    _reset_all_joints(env, env_ids, joint_pos, joint_vel)


def base_ang_vel_yaw_l2_stationary(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for yaw spinning while command velocity is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_b = quat_rotate_inverse(asset.data.root_link_quat_w, asset.data.root_ang_vel_w)
    return torch.square(ang_b[:, 2])

def penalty_arm_roll_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    手臂防穿模机制：如果 zarm_.*_joint 过于偏离初始姿态（如大角度反卷），给予严厉惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 动态抓取含有 'zarm' 关键字的关节索引
    joint_names = asset.joint_names
    arm_indices = [i for i, name in enumerate(joint_names) if "zarm" in name]
    
    if not arm_indices:
        return torch.zeros(env.num_envs, device=env.device)
        
    arm_joint_pos = asset.data.joint_pos[:, arm_indices]
    default_arm_pos = asset.data.default_joint_pos[:, arm_indices]
    
    # 若关节偏离初始安全姿态超过 1.0 弧度，开始进行平方惩罚，防止双手交叉打结
    diff = torch.abs(arm_joint_pos - default_arm_pos)
    violation = torch.clamp(diff - 1.0, min=0.0)
    
    return torch.sum(torch.square(violation), dim=1)

# ==============================================================================
# 以下为从 old_rewards.py 完整迁移过来的基础运动及平衡奖励函数
# ==============================================================================

def feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_air_time_clip(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold_min: float, threshold_max: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    air_time = (last_air_time - threshold_min) * first_contact
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_air_time_positive_biped(env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_slide(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0)
    asset : Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def joint_power_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_power = (asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.sum(torch.abs(joint_power), dim=1)

class action_smoothness_l2(ManagerTermBase):
    def __init__(self, env: ManagerBasedEnv, cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        super().__init__(env, cfg)
        self.prev_prev_action = None

    def __call__(self, env: ManagerBasedEnv, cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        if self.prev_prev_action is None:
            self.prev_prev_action = env.action_manager.prev_action.clone()
        action_smoothness_l2 = torch.sum(torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + self.prev_prev_action), dim=1)
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return action_smoothness_l2

def base_height_l2(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(dim=-1)
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    base_height = torch.nan_to_num(base_height, nan=target_height, posinf=target_height, neginf=target_height)
    return torch.square(base_height - target_height)

def track_lin_vel_xy_yaw_frame_exp(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_world_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, violation_max: float = torch.inf) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum(violation.clip(min=0.0, max=violation_max), dim=1)

def stand_still_without_cmd(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=-1)
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1)
    return reward

def gravity_aligned_when_stopping(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    is_zero_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.05
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat = asset.data.root_link_quat_w
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    pitch = torch.asin(2.0 * (w * y - x * z))
    reward = torch.exp(-5.0 * torch.square(pitch))
    masked_reward = torch.zeros_like(reward)
    masked_reward[is_zero_cmd] = reward[is_zero_cmd]
    return masked_reward