# 🚀 Kuavo 4 Pro 具身智能二次开发实战作品集

本仓库记录了在乐聚 **Kuavo 4 Pro（四代机 / S49）** 上进行二次开发的完整实战笔记、魔改代码与实验归档。

从 **Isaac Lab 强化学习 Sim2Sim / Sim2Real**，到 **NUC + Orin 双机 VLA 语音抓取、MoveIt 视觉避障、Gemini 全双工对话、人脸识别与视觉跟随**，再到 **TD-MPC2 世界模型探索**——这里沉淀的是官方文档很少覆盖的**终端命令、踩坑复盘、参数缝合思路**，以及那些在真机上反复炸机后才摸清的底层排障逻辑。

> 通用机器人知识见同级目录 [`robotics-notes`](../robotics-notes/)。

---

## ⚠️ 免责声明 (Disclaimer)

**有必要。** 本仓库大量内容基于乐聚官方开源生态，请务必阅读以下条款：

* **版权与知识产权：** 乐聚 Kuavo 机器人相关的底层 C++ 控制框架、OCS2/WBC 架构、官方 URDF/USD 模型、ROS 功能包及原始算法逻辑的知识产权均归 **[乐聚机器人 (Leju Robot)](https://gitee.com/leju-robot)** 及其官方开源仓库所有。包括但不限于：
  * [`kuavo-ros-opensource`](https://github.com/LejuRobotics/kuavo-ros-opensource)（下位机）
  * [`kuavo_ros_application`](https://github.com/LejuRobotics/kuavo_ros_application)（上位机）
  * [`leju_robot_rl`](https://gitee.com/leju-robot/leju_robot_rl)（Isaac Lab 训练）
  * [`kuavo-rl-opensource`](https://gitee.com/leju-robot/kuavo-rl-opensource)（RL 部署）
* **本仓库定位：** 仅为作者个人学习、调试与复盘所整理的**笔记、配置片段与自建 Demo 代码**，**不构成**乐聚官方文档的替代，**不代表**乐聚官方立场，**不保证**与最新官方版本兼容。请始终配合官方仓库与最新发行版结合使用。
* **代码边界：** 本仓库中的 `kuavo-ros-opensource/`、`kuavo_ros_application/` 等为**魔改影子目录**（非完整官方包）；`leju_robot_rl/`、`kuavo-rl-opensource/` 等含官方 fork 与作者改动并存。子目录内 ocs2、docker 等附带的 README 为上游原文。
* **禁止误用：** 未经乐聚官方授权，请勿将本仓库内容用于商业交付、OEM 贴牌或任何暗示「官方认证」的用途。
* **真机安全警告：** 人形机器人真机调试具有极高物理风险（关节爆冲、摔机、电池甩落等）。参考本仓库进行点火实测前，**必须**做好龙门架防坠、急停链路验证、安全员隔离与低速空载测试。**作者不对因参考本仓库内容造成的设备损坏或人身伤害承担任何责任。**

---

## 💻 硬件与阅读门槛 (Hardware & Prerequisites)

笔记按难度分层标注，便于按需选读：

| 标记 | 含义 |
|------|------|
| 🟢 | 纯仿真 / 代码阅读，**无需真机**（需 Ubuntu + 独立显卡用于 Isaac Lab） |
| 🟡 | 需 **Orin 上位机** 或 **NUC 下位机** 之一（视觉 / 大模型 / 单臂等） |
| 🔴 | **强依赖完整 Kuavo 4 Pro 真机**（双机联调、RL 真机部署、全身动作等） |

* 🟢 **RL 训练与 Sim2Sim（15.1 – 15.3、23.3 – 23.4）**：带 NVIDIA GPU 的 Ubuntu 主机即可，从 Isaac Lab 炼丹到 MuJoCo 闭环，无需机器人。
* 🟡 **视觉 / VLA / 大模型（4.x、21.x、22.x、30、32）**：至少需要 Orin NX 或 NUC 其中一侧；完整抓取需双机组网。
* 🔴 **RL Sim2Real、MoveIt 真机抓取、舞蹈部署（15.4、23.5 – 23.6、28）**：需要 Kuavo 4 Pro 物理真机 + 龙门架 + 熟练急停操作。

**双机架构（真机联调时）：**

| 角色 | 硬件 | 工作空间 |
|------|------|----------|
| 下位机 | Intel NUC | `~/kuavo-ros-opensource` |
| 上位机 | Jetson Orin NX | `~/kuavo_ros_application` |

---

## 📚 核心知识库目录 (Documentation)

完整索引见 [`kuavo_notes/README.md`](./kuavo_notes/README.md)。建议按主题链路阅读：

### 🟢 入门与环境

| 笔记 | 说明 |
|------|------|
| [`1.start.md`](./kuavo_notes/1.start.md) | 仿真 / 实机环境部署 |
| [`2.first_node.md`](./kuavo_notes/2.first_node.md) | 第一个 ROS 节点 |
| [`接口使用文档.md`](./kuavo_notes/接口使用文档.md) | SDK 接口速查 |
| [`25.update.md`](./kuavo_notes/25.update.md) | 官方包升级冲突与 launch 排障 |

### 🟡 感知、抓取与 VLA

| 笔记 | 说明 |
|------|------|
| [`4.3`](./kuavo_notes/4.3.real_robot_yolo_environment.md) · [`4.4`](./kuavo_notes/4.4real_visual_grasp.md) | 真机 YOLO 环境与 TF2 视觉抓取 |
| [`22.1VLA_grasping.md`](./kuavo_notes/22.1VLA_grasping.md) | VLA 语音抓取（双机 9 终端） |
| [`22.2.tree_VLA_grasp.md`](./kuavo_notes/22.2.tree_VLA_grasp.md) | 行为树版 VLA（`py_trees`） |
| [`28.moveit_grasping.md`](./kuavo_notes/28.moveit_grasping.md) | MoveIt 经典版 / OctoMap 双轨抓取 |
| [`20.gripper_issue.md`](./kuavo_notes/20.gripper_issue.md) | 夹爪安全（`claw_safe.py`） |

### 🟡 大模型、语音与人脸

| 笔记 | 说明 |
|------|------|
| [`21.2.local_AI_large_model.md`](./kuavo_notes/21.2.local_AI_large_model.md) | 离线 / 局域网大模型语音交互 |
| [`21.3.gemini_model.md`](./kuavo_notes/21.3.gemini_model.md) | Gemini 全双工跨国网关 |
| [`30.AI_image_identification.md`](./kuavo_notes/30.AI_image_identification.md) | VLM 图像触发 |
| [`32.1`](./kuavo_notes/32.1.face_recognition.md) · [`32.2`](./kuavo_notes/32.2.face_recognition_traking.md) | 人脸识别 + 跟踪融合 |
| [`24.1.visual_tracking.md`](./kuavo_notes/24.1.visual_tracking.md) | 头身协同视觉跟随 |

### 🟢 → 🔴 强化学习：行走 Sim2Sim / Sim2Real

| 笔记 | 标记 | 说明 |
|------|------|------|
| [`15.1.RL_lab_train.md`](./kuavo_notes/15.1.RL_lab_train.md) | 🟢 | Isaac Lab 训练全流程 |
| [`15.2RL_lab_analysis_code.md`](./kuavo_notes/15.2RL_lab_analysis_code.md) | 🟢 | 奖励函数 / 域随机化拆解 |
| [`15.3RL_lab_sim2sim.md`](./kuavo_notes/15.3RL_lab_sim_to_sim.md) | 🟢 | MuJoCo Sim2Sim 闭环 |
| [`15.4RL_lab_sim_to_real.md`](./kuavo_notes/15.4RL_lab_sim_to_real.md) | 🔴 | **真机部署黑皮书**：v42/v46/v49 参数缝合、WBC 起立爆冲、50Hz/1000Hz 线程咬合 |

### 🟢 → 🔴 强化学习：S49 全身舞

| 笔记 | 标记 | 说明 |
|------|------|------|
| [`23.1.RL_dance_overview.md`](./kuavo_notes/23.1.RL_dance_overview.md) | 🟢 | 舞蹈 RL 总览与分支纪律 |
| [`23.3.RL_dance_train.md`](./kuavo_notes/23.3.RL_dance_train.md) | 🟢 | S49 训练（115 维 obs、mimic 奖励） |
| [`23.4.RL_dance_sim2sim.md`](./kuavo_notes/23.4.RL_dance_sim2sim.md) | 🟢 | MuJoCo 舞蹈验证 |
| [`23.5`](./kuavo_notes/23.5.RL_dance_deploy_hybrid.md) · [`23.6`](./kuavo_notes/23.6.RL_dance_terminal_commands.md) | 🔴 | 真机混合部署与终端命令全集 |

### 🟢 世界模型

| 笔记 | 说明 |
|------|------|
| [`31.1.world_model.md`](./kuavo_notes/31.1.world_model.md) | TD-MPC2 / `leju_robot_wm` 实验 |

### 📎 官方案例（参考）

[`5功能案例/案例目录.md`](./kuavo_notes/5功能案例/案例目录.md) — 乐聚官方功能案例索引（本仓库未改写）。

---

## 💡 为什么会有这个仓库？

人形机器人的二次开发绝不是「克隆官方仓库 → 改两行 launch → 点火」。在 Kuavo 4 Pro 上，我们遭遇的是**跨版本、跨机器、跨框架**的系统性摩擦：

* **官方未完全适配 S49 的全链路：** 行走 RL 面向 S42/S46；舞蹈需自建 S49 任务树；部署包骨架、训练资产与 v49 真机物理参数存在「三体撕裂」，需手动缝合 URDF、`.info` 与 `humanoidController.cpp`。
* **双机架构的认知负担：** NUC 跑 WBC/IK/MoveIt，Orin 跑 YOLO/ASR/大模型——路径、权限、代理与 ROS 话题的跨机对齐，官方文档往往默认单机。
* **官方升级覆盖魔改：** `load_kuavo_real.launch`、`kuavo.json` 等在版本升级中被还原，需 git stash / 手动合并（见 25.update）。
* **Sim2Real 起立阶段的物理灾难：** 经典 WBC 起立与 RL 切换之间的阶跃、刚度与触地滤波问题，可直接导致爆冲与硬件损伤（见 15.4）。
* **文档分散、深坑未记：** 上述问题的排障过程分散在数百次终端会话中；本仓库的价值是**拒绝重复踩坑**，把「现象 → 根因 → 命令 → 代码」串成可复现的笔记。

---

## 📦 代码归档 (Code Snapshot)

本仓库同时归档**魔改代码影子目录**（非官方完整包），与 `kuavo_notes/` 文档交叉引用：

```
kuavo-dev-notes/
├── kuavo_notes/              # 📖 实战文档（本 README 的知识库主体）
├── kuavo-ros-opensource/     # 🟦 下位机魔改（VLA / MoveIt / teleop / 资产配置）
├── kuavo_ros_application/    # 🟩 上位机魔改（YOLO / 大模型 / 人脸 / 视觉跟随）
├── leju_robot_rl/            # 🧪 Isaac Lab RL 训练（含 S49 舞蹈）
├── leju_robot_wm/            # 🧪 TD-MPC2 世界模型
├── kuavo-rl-opensource/      # 🤖 RL 部署（MuJoCo / 真机 Docker）
└── automove_and_link.sh      # 本机 ~/kuavo_all 软链接辅助脚本（可选）
```

各目录说明见对应 `README.md`。

**说明：** 真机源码主要在 NUC / Orin 上维护；本仓库为开发机上的**作品集快照**。真机不会随 `git push` 自动变更。

---

## 🛠️ 技术栈 (Tech Stack)

* **系统：** Ubuntu 20.04 / 22.04 LTS，Docker（RL 部署）
* **机器人中间件：** ROS Noetic，catkin
* **仿真与 RL：** Isaac Sim 4.2，Isaac Lab，RSL-RL (PPO)，MuJoCo 3.x
* **世界模型：** TD-MPC2（`leju_robot_wm`）
* **视觉：** YOLOv8，OpenCV，TF2，MoveIt / OctoMap
* **具身交互：** Faster-Whisper，Gemini API，ONNX Runtime（人脸）
* **部署运行时：** C++14，Python 3，ONNX Runtime，OCS2，WBC

---

## 📤 推送 GitHub 建议

* 全仓本地约 **26GB**（含训练日志与编译产物），**请勿整包 push**。
* **作品集推荐：** `kuavo_notes/` + `kuavo-ros-opensource/` + `kuavo_ros_application/` + 根 README（约 100MB 级）。
* 推送 RL/WM/部署整仓时，务必 `.gitignore` 排除 `logs/`、`outputs/`、`build/`、`devel/`、`*.pt`、`*.onnx` 等大文件。

---

## 🤖 关于作者 (About)

机器人工程专业毕业生，有人形机器人二次开发经验、轮式机器人设计经验。目前正致力于探索 AI 驱动的具身智能控制前沿，即将在机器人与人工智能领域开启进一步的硕士研究。热衷于跨越仿真与真实物理世界之间的鸿沟，并把那些「官方文档不会写」的踩坑过程整理成可复现的笔记。

---

## 🔗 官方 upstream（请优先 star / fork 官方）

| 仓库 | 链接 |
|------|------|
| 下位机开源 | https://github.com/LejuRobotics/kuavo-ros-opensource |
| 上位机应用 | https://gitee.com/leju-robot/kuavo_ros_application |
| RL 训练 | https://gitee.com/leju-robot/leju_robot_rl |
| RL 部署 | https://gitee.com/leju-robot/kuavo-rl-opensource |

---

<p align="center"><sub>本仓库为个人学习笔记与作品集，与乐聚机器人无官方关联。</sub></p>
