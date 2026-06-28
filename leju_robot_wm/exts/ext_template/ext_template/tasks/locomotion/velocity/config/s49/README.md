> 世界模型仓内副本。TD-MPC2 见 [`kuavo_notes/31.1`](../../../../../../../../kuavo_notes/31.1.world_model.md)。下列 S49 舞蹈流程与 `leju_robot_rl` 同目录 README 一致，供 WM 实验参考。

# S49 四代机全身舞 — 训练 / 导出 / 部署

与 `config/s42/` **并行**，不修改原有 S42/S46 训练链路。训练、MuJoCo、真机统一使用 **ROBOT_VERSION=49**。

## 1. 准备 Isaac 资产

```bash
cd leju_robot_rl
bash scripts/tools/setup_s49_training_assets.sh
```

脚本会从 `../kuavo-ros-opensource/.../biped_s49` 软链到 `assets/Robots/Kuavo/biped_s49`，并生成：

- `biped_s49_rl.urdf` — mesh 相对路径
- `biped_s49_26dof.urdf` — 冻结手指/头 22 关节 → 26 revolute
- `biped_s49_26dof_lite.urdf` — **训练默认**：去掉手指/相机/头碰撞体（对齐 S46 USD 简化碰撞）

## 2. 训练

**重要**：S49 与 S46 不同，没有官方优化 USD。训练链路已做以下适配（无需改 S42/S46）：

| 问题 | 修复 |
|------|------|
| S42 踝解算器在 S49 URDF 上 NaN | `DelayedPDActuator_S49` 关节空间 PD |
| 手指/相机碰撞拖慢 PhysX | `biped_s49_26dof_lite.urdf` |
| 真实力矩上限过低站不稳 | 训练暂用 S46 等效 effort limit |
| 躺地/跪地模仿 CSV 局部最优 | 姿态+高度门控 `track_punch_*_upright` + `penalty_root_squat`（**防跪 v2**） |
| 勾脚踮脚、仅后跟着地 | LAFAN1 G1 参考轨迹 + `penalty_foot_pitch`（**防勾脚 v3**） |

**一键重训（推荐）**：LAFAN1 舞蹈 CSV 批量转换 + 训练

```bash
cd leju_robot_rl
conda activate isaaclab   # 或你的 Isaac Lab 环境
bash scripts/tools/run_s49_lafan1_retrain.sh
```

仅转换 LAFAN1 舞蹈（不训练）：

```bash
bash scripts/tools/batch_convert_lafan1_g1_dance.sh
# 输出: motion_refs/lafan1_g1/dance*_INPLACE_RAD.csv
# 训练默认: kuavo_action_LAFAN1_g1_dance1_INPLACE_RAD.csv
```

手动训练（8GB 卡请用 `--num_envs 4096`，仍 OOM 再降至 2048）：

```bash
bash scripts/tools/setup_s49_training_assets.sh
rm -rf /tmp/IsaacLab/usd_*
python3 scripts/rsl_rl/train.py \
  --task Legged-Isaac-Velocity-Flat-Kuavo-S49-Punch-v0 \
  --headless --num_envs 4096
```

TensorBoard 健康指标（iter ~500+）：

- `dof_pos_illegal` → 接近 0
- `flat_orientation_l2` → 趋近 0（负奖励变小）
- `track_punch_*` 在站直后才开始上升

实验日志目录：`logs/rsl_rl/Kuavo/s49/dance/`

## 3. 导出 ONNX

```bash
python scripts/rsl_rl/play.py \
  --task Legged-Isaac-Velocity-Flat-Kuavo-S49-Play-v0 \
  --load_run <run_name> \
  --export_onnx
```

将导出的 `policy.onnx` 重命名为 `policy_s49.onnx`，放入部署目录：

`../kuavo-rl-opensource/kuavo-robot-deploy/.../model/networks/policy_s49.onnx`

## 4. MuJoCo sim2sim

工作空间需包含 `kuavo-ros-opensource` 的 `kuavo_assets`（含 `biped_s49/xml/scene_rl.xml`）。

```bash
export ROBOT_VERSION=49
roslaunch humanoid_controllers load_kuavo_mujoco_sim_dance_s49.launch
```

或手动指定 RL 参数：

```bash
export ROBOT_VERSION=49
roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch \
  rl_param:=$(rospack find humanoid_controllers)/config/kuavo_v49/rl/skw_rl_param_dance.info
```

## 5. 真机

- 使用 `kuavo_v49/rl/skw_rl_param_dance.info`（115 维 obs，与训练一致）
- `reference.info` 等物理参数仍可按 `kuavo_notes/15.4RL_lab_sim_to_real.md` 从真机 bag 标定
- 统一 S49 后**不再需要** v46 骨架 + v49 参数的「移花接木」，但 com 高度、滤波、手臂 Kp 等 sim2real 经验仍然适用

## 文件对照

| 环节 | S42/S46（保留） | S49（新增） |
|------|----------------|-------------|
| 训练配置 | `config/s42/punch_env_cfg.py` | `config/s49/punch_env_cfg.py` |
| 机器人资产 | `Kuavos46_CFG` (USD) | `Kuavos49_CFG` (URDF) |
| Gym 任务 | `Legged-Isaac-Velocity-Flat-Kuavo-Punch-v0` | `Legged-Isaac-Velocity-Flat-Kuavo-S49-Punch-v0` |
| 部署 RL 参数 | `kuavo_v46/rl/skw_rl_param.info` | `kuavo_v49/rl/skw_rl_param_dance.info` |
| 策略文件 | `policy_s42.onnx` | `policy_s49.onnx` |
