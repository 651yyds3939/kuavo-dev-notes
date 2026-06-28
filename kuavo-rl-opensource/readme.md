# kuavo-rl-opensource — RL 部署

本目录为 [`kuavo-dev-notes`](../README.md) 本仓库的一部分，对应乐聚 **Kuavo RL 部署**仓库（MuJoCo 仿真 + 真机 EtherCAT）。

---

## 开发状态（给克隆者）

部署仓随 [`leju_robot_rl`](../leju_robot_rl/) 训练侧**同步迭代**：`humanoidController.cpp`、`.info` 与 `policy_*.onnx` 需与当前训练 run 对齐。**仓库内 ONNX 一般不保证为最新最优 checkpoint**；请自行训练导出后再拷贝至 `kuavo-robot-deploy/.../networks/`。Sim2Sim / 真机步骤见 [`kuavo_notes/15.4`](../kuavo_notes/15.4RL_lab_sim_to_real.md)、[`23.5–23.6`](../kuavo_notes/23.5.RL_dance_deploy_hybrid.md)。

## 结构

| 子目录 | 说明 |
|--------|------|
| `kuavo-robot-train/` | 旧版 Gym 训练（可选） |
| `kuavo-robot-deploy/` | **主部署工作空间**（catkin，真机 / MuJoCo） |

---

## 文档

| 主题 | 文档 |
|------|------|
| Lab 行走部署 | [`kuavo_notes/15.4`](../kuavo_notes/15.4RL_lab_sim_to_real.md) |
| S49 舞蹈部署 | [`kuavo_notes/23.4–23.6`](../kuavo_notes/23.4.RL_dance_sim2sim.md) |
| 部署操作 | [`kuavo-robot-deploy/readme.md`](./kuavo-robot-deploy/readme.md) |

---

## 关键改动（S49 舞蹈）

- `kuavo-robot-deploy/src/humanoid-control/humanoid_controllers/src/humanoidController.cpp`
- `config/kuavo_v49/rl/skw_rl_param_dance.info`
- `model/networks/policy_s49.onnx`（需自行训练导出，一般不 push）

真机/Docker 内路径通常为 `/root/kuavo_ws`（挂载本目录或同级副本）。

---

## 编译（deploy）

```bash
cd kuavo-rl-opensource/kuavo-robot-deploy
source installed/setup.bash   # 或官方指引
catkin build humanoid_controllers
```

---

## 推送 GitHub

排除 `build/`、`devel/`、`installed/`（若可重建）、大体积 `*.onnx`。子目录 ocs2 / docker 等 README 为上游原文。
