# leju_robot_wm — TD-MPC2 世界模型

本目录为 [`kuavo-dev-notes`](../README.md) 本仓库（`kuavo-dev-notes`）的一部分，在 `leju_robot_rl` 同构框架上扩展 **TD-MPC2** 世界模型实验（行走、舞蹈等任务配置）。

---

## 开发状态（给克隆者）

**当前结论：** TD-MPC2 在 Isaac Lab 上的 **train → TensorBoard → play 录视频** 流水线已跑通；主文档 [`kuavo_notes/31.1`](../kuavo_notes/31.1.world_model.md)。

**尚未完成：** 世界模型在舞蹈（`tdmpc2_dance`）与行走（`tdmpc2_velocity`）任务上的**长期稳定性与策略质量仍未达预期**，超参与奖励仍在试验阶段；**暂不以真机部署为目标**，以仿真研究为主。

**你可以在此基础上：**

- 改 `exts/.../world_model_core/` 与任务配置中的奖励、规划 horizon、batch 等
- 参照 31.1 调整 CSV / 参考动作与 `scripts/tdmpc2/train.py` 参数
- 对比 [`leju_robot_rl`](../leju_robot_rl/) 中已较成熟的 PPO 路线，按需迁移资产或奖励思路

日志目录：`logs/tdmpc2/Kuavo/s42/...`（本地可能通过软链接外置，见根目录 [`SYMLINK_LAYOUT.md`](../SYMLINK_LAYOUT.md)）。


---

## 文档

- 主文档：[`kuavo_notes/31.1.world_model.md`](../kuavo_notes/31.1.world_model.md)
- 代码说明：[`readme.md`](./readme.md)
- S49 配置：[`exts/.../config/s49/README.md`](./exts/ext_template/ext_template/tasks/locomotion/velocity/config/s49/README.md)（若存在）

---

## 环境

- Conda 环境：`isaaclab_wm`（见 31.1 文档）
- Isaac Sim 4.2 + Isaac Lab

```bash
cd leju_robot_wm
conda activate isaaclab_wm
pip install -e exts/ext_template
```

---

## 快速入口

```bash
cd leju_robot_wm
# 训练 / play 命令见 kuavo_notes/31.1.world_model.md
python3 scripts/tdmpc2/play.py  # 示例入口，参数见文档
```

日志：`logs/tdmpc2/Kuavo/...`

---

## 推送 GitHub

排除 `logs/`、`outputs/`、`*.pt` 等大文件。

---

## 与 leju_robot_rl 的区别

| | leju_robot_rl | leju_robot_wm |
|--|---------------|---------------|
| 算法 | PPO (RSL-RL) | TD-MPC2 |
| 典型用途 | 行走、S49 舞蹈 mimic | 世界模型 / 规划实验 |
| 部署 | kuavo-rl-opensource | 实验为主，见 31.1 |
