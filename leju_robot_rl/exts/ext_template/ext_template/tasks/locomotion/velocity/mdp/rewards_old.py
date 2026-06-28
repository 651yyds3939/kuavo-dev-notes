from __future__ import annotations

import torch
import math
import os
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedEnv

# 全局缓存动作特征，防止每次步进都重复读取CSV影响帧率
_PUNCH_TRAJECTORY_CACHE = {}

def _load_csv_to_cache(csv_path: str, device: torch.device):
    """辅助函数：将CSV加载进GPU显存，并自动对齐截断尾部冗余"""
    if csv_path not in _PUNCH_TRAJECTORY_CACHE:
        full_csv_path = csv_path
        if not os.path.isabs(full_csv_path):
            full_csv_path = os.path.abspath(full_csv_path)
        with open(full_csv_path, 'r') as f:
            lines = f.readlines()
        traj = [[float(x) for x in line.strip().split(',') if x] for line in lines if line.strip()]
        min_len = min(len(row) for row in traj)
        traj = [row[:min_len] for row in traj]
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

def track_punch_joint_trajectory_exp(
    env: ManagerBasedRLEnv,
    std: float,
    csv_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    核心魔改：读取CSV参考轨迹并计算关节位置的指数惩罚跟随奖励。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 1. 从缓存获取轨迹
    traj_tensor = _load_csv_to_cache(csv_path, env.device)
    num_frames = traj_tensor.shape[0]
    
    # 2. 根据 Episode current step 计算当前对应的CSV帧
    if not hasattr(env, "episode_length_buf"):
        frame_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        frame_indices = (env.episode_length_buf % num_frames).long()
    target_joint_pos = traj_tensor[frame_indices]
    
    # 3. 对齐关节数量进行均方误差计算
    num_target_joints = target_joint_pos.shape[1]
    max_joints = min(num_target_joints, asset.data.joint_pos.shape[1])
    
    current_joint_pos = asset.data.joint_pos[:, :max_joints]
    target_joint_pos = target_joint_pos[:, :max_joints]
    
    # 🔴 使用 mean 求全身关节的平均误差，防止多关节的微小误差累加导致指数函数极小化爆炸
    joint_pos_error = torch.mean(torch.square(current_joint_pos - target_joint_pos), dim=1)
    return torch.exp(-joint_pos_error / std**2)

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