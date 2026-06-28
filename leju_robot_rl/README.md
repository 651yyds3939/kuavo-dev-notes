# leju_robot_rl — Isaac Lab 强化学习训练

本目录为 [`kuavo-dev-notes`](../README.md) 本仓库（`kuavo-dev-notes`）的一部分，基于乐聚 **[leju_robot_rl](https://gitee.com/leju-robot/leju_robot_rl)**（Isaac Lab + RSL-RL）fork，含 **S49 四代机全身舞** 等自定义任务。

---

## 开发状态（给克隆者）

**当前结论：** 行走（S42 flat/rough）与 S49 全身舞蹈的 **Sim2Sim → 训练 → 导出 ONNX → 部署验证** 链路已在作者环境中跑通；文档见 [`kuavo_notes/15.*`](../kuavo_notes/15.1.RL_lab_train.md)、[`23.*`](../kuavo_notes/23.1.RL_dance_overview.md)。

**尚未完成：** 策略在仿真与真机上的**稳定性与观感仍未达到作者预期**（尤其 S49 舞蹈 mimic 质量、部分行走 checkpoint）。奖励权重、域随机化、动作 CSV 与 S49 任务配置**仍在快速改动**，提交历史可能较乱。

**你可以在此基础上：**

- 改 `exts/.../mdp/rewards.py` 与各 `*env_cfg.py` 中的 `RewTerm` / `weight`
- 替换或新增根目录 `kuavo_action_*.csv`（见 [`23.2`](../kuavo_notes/23.2.RL_dance_motion_data.md)）
- 用 `scripts/rsl_rl/train.py` / `play.py` 与 `logs/rsl_rl/...` 自行炼丹；勿期望仓库内某一版 checkpoint 即为「最终成品」

上游 fork 壳来自乐聚官方 [`leju_robot_rl`](https://gitee.com/leju-robot/leju_robot_rl)；S49 舞蹈等为作者扩展，与官方默认任务不同。


---

## 与本仓库的关系

| 资源 | 路径 |
|------|------|
| 舞蹈训练文档 | [`kuavo_notes/23.*`](../kuavo_notes/23.1.RL_dance_overview.md) |
| 行走 Lab 文档 | [`kuavo_notes/15.*`](../kuavo_notes/15.1.RL_lab_train.md) |
| 代码协作说明 | [`readme.md`](./readme.md)（MDP / PPO 流水线） |
| S49 任务说明 | [`exts/.../config/s49/README.md`](./exts/ext_template/ext_template/tasks/locomotion/velocity/config/s49/README.md) |
| 部署仓 | [`../kuavo-rl-opensource`](../kuavo-rl-opensource/) |
| S49 URDF 资产 | 同级 [`../kuavo-ros-opensource`](../kuavo-ros-opensource/) 或真机 `kuavo_assets` |

---

## 环境

- **Isaac Sim 4.2** + **Isaac Lab**
- Conda 环境：`isaaclab`（见官方 / 乐聚安装文档）
- Git LFS（`*.onnx` 等大文件）

```bash
cd leju_robot_rl
git lfs install
pip install -e exts/ext_template
```

---

## S49 全身舞（快速入口）

```bash
cd leju_robot_rl
conda activate isaaclab
bash scripts/tools/setup_s49_training_assets.sh
bash scripts/tools/run_s49_lafan1_retrain.sh   # 转换 CSV + 训练
```

TensorBoard：`logs/rsl_rl/Kuavo/s49/dance/`  
导出 ONNX 后拷贝至 `kuavo-rl-opensource/kuavo-robot-deploy/.../networks/`。

细节见 **s49/README.md** 与 **kuavo_notes/23.3–23.6**。

---

## 推送 GitHub

勿提交 `logs/`、`outputs/`、`.hydra/` 缓存及大 checkpoint。训练产物仅保留说明性路径即可。

---

## 上游

根目录原 Isaac Lab 模板说明已替换为本仓库导读；安装细节请参考乐聚 Gitee 官方仓库与 NVIDIA Isaac Lab 文档。
