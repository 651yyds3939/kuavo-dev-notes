# leju_robot_wm — TD-MPC2 世界模型代码说明

> **本目录说明**：本仓为 [`kuavo-dev-notes`](../README.md) 子目录。操作手册见 [`kuavo_notes/31.1.world_model.md`](../kuavo_notes/31.1.world_model.md)。

---

## 一、整体逻辑

在 Isaac Lab 环境上接入 **TD-MPC2**（Temporal Difference Model Predictive Control），用学习到的世界模型做短 horizon 规划，与 `leju_robot_rl` 的 PPO 路线并行实验。

- **环境配置**：`exts/ext_template/.../config/s42/`、`s49/` 等（rough / punch / velocity）
- **训练脚本**：`scripts/tdmpc2/`（如 `train.py`、`play.py`）
- **日志**：`logs/tdmpc2/Kuavo/...`

---

## 二、与 leju_robot_rl 的关系

| 项目 | leju_robot_rl | leju_robot_wm |
|------|---------------|---------------|
| 算法 | RSL-RL PPO | TD-MPC2 |
| Conda 环境 | `isaaclab` | `isaaclab_wm` |
| 主要文档 | kuavo_notes/15.*、23.* | kuavo_notes/31.1 |
| 部署 | kuavo-rl-opensource | 以仿真 / 研究为主 |

资产准备（S49 URDF 等）可复用 `leju_robot_rl/scripts/tools/setup_s49_training_assets.sh` 的思路；详见 31.1。

---

## 三、常用路径

```
leju_robot_wm/
├── scripts/tdmpc2/          # 训练与 play
├── exts/ext_template/       # 任务 / 奖励 / TD-MPC2 agent 配置
├── logs/tdmpc2/             # 实验日志（勿 push）
└── outputs/                 # Hydra 输出（勿 push）
```

---

## 四、推送 GitHub

排除 `logs/`、`outputs/`、`*.pt` 及 Isaac 缓存。代码与配置可随本仓库提交。
