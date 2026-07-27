# 给 AI 的奖励函数设计任务

## 背景

我们在做人形机器人 Kuavo S49（26 自由度，12 腿 + 14 臂）的**原地舞蹈强化学习**。技术栈：Isaac Lab 1.4.1 + Isaac Sim 4.2.0 + PPO (rsl_rl)，8GB 显存 GPU，4096 并行环境。

**当前目标**：先做一个**纯上半身手臂跟踪**版本，机器人站稳不动腿，只挥动手臂跟踪 CSV 舞蹈轨迹。这个版本通过后部署到真机验证，再回头攻克全身舞蹈。

## 关键文件路径

```
训练包根目录: /home/lwy/Notes/kuavo-dev-notes/leju_robot_rl/

奖励函数代码: exts/ext_template/ext_template/tasks/locomotion/velocity/mdp/rewards.py
环境配置:     exts/ext_template/ext_template/tasks/locomotion/velocity/config/s49/punch_env_cfg.py
训练脚本:     scripts/tools/run_s49_lafan1_retrain.sh
配置生成:     scripts/tools/write_run_config.py

训练日志:     logs/rsl_rl/Kuavo/s49/dance/<timestamp>/
  ├── RUN_CONFIG.md          ← 包含奖励权重+最终收敛值+训练结果描述
  ├── params/env.yaml        ← 完整环境配置快照
  └── events.out.tfevents.*  ← TensorBoard 日志

动作CSV:      kuavo_action_S49_FROM_S54_INPLACE_RAD.csv  (1565帧, 26列弧度值)
CSV关节顺序:  leg_l1..l6, leg_r1..r6, zarm_l1..l7, zarm_r1..r7
CSV第1帧站姿: 膝~0.52rad(30°), 踝~-0.30rad(-17°), 髋~-0.27rad(-15°)
```

## 17 次训练完整历史

你可以看每次训练的 RUN_CONFIG.md 包含：奖励权重配置 + TensorBoard 最终收敛值 + 训练结果。

| Run ID | 版本 | 结果 | 根因 |
|--------|------|------|------|
| 2026-06-18 | v1 | 出生就倒下 | 弱稳定(flat=-1)，手臂无upright门控 |
| 2026-06-20 | v1.5 | 跪着跳舞 | 有门控但缺squat惩罚 |
| 2026-06-26 | v2 | **勾脚跳舞** ← 最佳版本 | flat=-12/squat=-15/arms=8.68/base_contact=1%，唯一缺的就是脚下约束 |
| 2026-06-27 | v3 | 手臂不动 | LAFAN1 CSV + foot_pitch=-6 + upright=0.88太严 |
| 2026-07-02 | v5 | 乱飘 | HYBRID CSV + 防漂移太弱 |
| 2026-07-03 | v6 | 乱飘 | base_lin_vel=-0.05太弱 |
| 2026-07-04 | v7 | 腿抖 | 加root_xy锚定但smoothness太弱 |
| 2026-07-05 | v8 | 勾脚跳舞 | smoothness够但foot_pitch=-2太轻 |
| 2026-07-06 | v9 | 出生就死 | foot_pitch=-8+foot_link_flat=-4+force_vel=-0.5太猛 |
| 2026-07-07 | v9.1 | 出生就死 | 脚下约束仍然太重 |
| 2026-07-07 | v10 | 跪着跳舞 | flat=-2/squat=-5太弱+手臂无upright门控 |
| 2026-07-08 | v11 | 勾脚+2秒倒 | flat=-5/squat=-10不够+加脚下约束致不稳定 |
| 2026-07-09 | v12 | 勾脚+站稳 | 回v2基线flat=-12/squat=-15+foot_pitch=-2，存活好但勾脚 |
| 2026-07-10 | v14 | 勾脚+腿颤 | 加foot_link_flat导致震颤 |
| 2026-07-11 | v15初版 | 存活短 | 中等稳定(flat=-5/squat=-8)无法引导站立 |
| 2026-07-12 | v15二版 | 出生跳一下即倒 | 腿锁-5.0太强+URDF直腿spawn |
| 2026-07-13 | v15三版 | 腿颤+秒倒 | 腿锁-5.0阻止平衡微调，v2级稳定但腿不能动 |

## v2（最佳版本）的关键数据

```
track_punch_arms: 8.68 (满分12.0, 72%跟踪质量)  ← 所有版本最高
track_punch_legs: 3.15 (满分5.0, 63%跟踪质量)
flat_orientation_l2: -0.032 ← 几乎完美直立
penalty_root_squat: ~0 ← 零下蹲
base_contact终止率: 1.0% ← 几乎从不摔倒
mean_reward: 83.6 ← 碾压其他版本

配置:
  flat_orientation=-12, base_height=-8, squat=-15, min_height=0.72
  arms=12.0, upright gate 0.88, std=0.38
  legs=5.0, upright gate 0.88, std=0.50
  arm_roll=-3.0, action_rate=-0.004, smoothness=-0.006
  无任何脚下约束

问题: 脚踝勾起来(脚跟支撑)，因为没有penalty_foot_pitch
```

## 当前 v15 配置（纯上半身）

```python
# 稳定性（v2 级别）
flat_orientation_l2: -12.0
base_height: -8.0, target=0.87
penalty_root_squat: -15.0, min_height=0.72

# 手臂跟踪
track_punch_arms: 15.0, std=0.38, upright_gate min_upright=0.75
track_punch_legs: 0.0   ← 砍掉腿部跟踪

# 腿部软锁（最近从 -5.0 降到 -0.5，允许平衡微调）
joint_deviation_legs: -0.5  (目标=CSV第1帧站姿: 膝30° 踝-17°)

# 防漂移
base_lin_vel_xy_stationary: -1.0
penalty_root_xy_displacement: -2.0
base_ang_vel_yaw_stationary: -0.3

# 轻阻尼（v2级别，不压制手臂动作）
action_rate_l2: -0.01
action_smoothness_l2: -0.01

# 无脚下约束
# 无arm_roll_penalty

# Spawn: 腿出生就在站姿(reset_legs_to_standing函数)，不是URDF零位直腿
```

## 核心教训

1. **门控型奖励需要极稳底座**：手臂奖励被upright gate锁住，必须先站起来才能拿分。flat/squat必须强到能独立引导站立(v2的-12/-15才行，-5/-8不行)。

2. **人形机器人手臂运动需要腿部微调配合**：腿锁太死(-5.0)阻止平衡补偿→手臂一动就倒。需要轻锁(-0.5)或让腿自由。

3. **脚下约束是双刃剑**：v2无脚下约束能站稳跳舞但勾脚。加foot_pitch/foot_link_flat太重(v9,-4/-8)直接致死，太轻(v8,-2)不起作用。

4. **奖励权重之间不是线性关系**：flat从-12降到-5不是"宽松一点"，而是彻底改变了最优解结构——机器人从"先站稳再跳舞"变成"直接放弃站立"。

5. **Spawn姿态很重要**：URDF默认关节角是全零(直腿)，但机器人自然站姿是膝30°。出生在错误姿态会直接摔倒。

## 你的任务

修改 `punch_env_cfg.py` 的 `RewardsCfg`，让机器人：
1. **站稳**（不跪、不倒、不勾脚）
2. **手臂跟踪 CSV 舞蹈轨迹**（幅度够大、跟拍子）
3. **腿保持自然站姿**（膝~30°，可微调维持平衡但不跳舞）

如果需要新增奖励函数，修改 `rewards.py`。修改完成后更新 `run_s49_lafan1_retrain.sh` 里的版本号注释。

训练命令：
```bash
cd ~/kuavo_all/leju_robot_rl && conda activate isaaclab
bash scripts/tools/run_s49_lafan1_retrain.sh
```
