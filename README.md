# 🚀 Kuavo 4 Pro 具身智能二次开发笔记与代码归档

本仓库记录了在乐聚 **Kuavo 4 Pro（四代机 / S49）** 上进行二次开发的完整实战笔记、魔改代码与实验归档。

从 **Isaac Lab 强化学习 Sim2Sim / Sim2Real**，到 **NUC + Orin 双机 VLA 语音抓取、MoveIt 视觉避障、Gemini 全双工对话、人脸识别与视觉跟随**，再到 **TD-MPC2 世界模型探索**——这里沉淀的是官方文档很少覆盖的**终端命令、踩坑复盘、参数缝合思路**，以及那些在真机上反复炸机后才摸清的底层排障逻辑。

> 通用机器人知识见独立仓库 [robotics-notes](https://github.com/651yyds3939/robotics-notes)（两仓库跨链请用 GitHub 地址，勿用 `../` 相对路径）。

> **整理状态：** 部分 `.md` 笔记与工作空间内的魔改程序**仍在撰写或整理中**，可能存在章节缺失、路径与当前真机不一致、或与最新官方版本未对齐等情况。请以文档内的终端命令与可运行代码为准；发现问题欢迎提 Issue。
>
> **训练效果：** [`leju_robot_rl/`](./leju_robot_rl/)（PPO 行走 / S49 舞蹈）与 [`leju_robot_wm/`](./leju_robot_wm/)（TD-MPC2 世界模型）的**策略效果尚未达到作者预期的稳定程度**，相关代码与奖励仍在快速迭代；**完整训练 → 验证 → 部署链路已跑通**，克隆者可在此基础上自行改奖励函数、替换 / 追加 CSV 动作数据等。详见各子目录 `README.md`。

---

## 持续更新与本地同步

本仓库**会持续更新**（`kuavo_notes/` 实战文档、魔改代码影子目录、RL 子仓等）。若已通过 `git clone` 拉取到本机，日后想获取 GitHub 上的最新内容，在**该仓库根目录**执行：

```bash
cd /path/to/kuavo-dev-notes   # 改为本机实际路径
git pull
```

若本地有未提交修改，`git pull` 可能提示冲突；可先 `git stash` 暂存改动，同步后再 `git stash pop`，或先提交到 fork 再拉取。

**两库并排维护时**（例如 `~/Notes/kuavo-dev-notes` 与 `~/Notes/robotics-notes`），需分别在两个目录各执行一次 `git pull`。通用机器人知识见 [robotics-notes](https://github.com/651yyds3939/robotics-notes)。

**首次克隆本仓库（含 Git LFS 须知）：**

本仓库在 `kuavo-ros-opensource/src/demo/vla_grasp/pretrained_model/` 下含约 **200MB** 的 ACT 预训练模型（`*.safetensors`），通过 **Git LFS** 托管。克隆前请先安装并初始化 LFS，否则只会拉到约 130 字节的指针文件，真机推理会失败。

```bash
# 1. 安装 Git LFS（任选其一，一次性）
conda install -c conda-forge git-lfs
# sudo apt install git-lfs

git lfs install

# 2. 克隆
git clone https://github.com/651yyds3939/kuavo-dev-notes.git
cd kuavo-dev-notes
```

**已 clone 但未装 LFS？** 进入仓库后补装并拉取大文件：

```bash
git lfs install
git lfs pull
```

**验证模型是否完整：**

```bash
ls -lh kuavo-ros-opensource/src/demo/vla_grasp/pretrained_model/model.safetensors
# 正常应约 198M；若仅 ~134 字节，说明仍是 LFS 指针，请执行 git lfs pull
```

**日常同步：**

```bash
cd /path/to/kuavo-dev-notes   # 改为本机实际路径
git pull
git lfs pull   # 若 pull 后新增了 LFS 大文件，建议执行
```


---

## 🔗 工作空间软链接与迁移（作者本机专用，克隆者可跳过）

> **给 GitHub 访客：** 下面两个文件是作者在自己电脑上的**目录整理脚本与说明**（`~/kuavo_all`、`~/Notes` 路径、`logs` 归档等）。 
> **clone 本仓库阅读笔记/代码时，通常无需运行**；只有当要复刻作者同一套「本仓库 + 终端入口 + 训练产物外置」目录布局时再看。

- **拓扑说明：** [`SYMLINK_LAYOUT.md`](./SYMLINK_LAYOUT.md) — 软链接三层结构、铁律与排障
- **维护脚本：** [`automove_and_link.sh`](./automove_and_link.sh) — 迁移/归档/验链（`./automove_and_link.sh help`）




## ⚠️ 免责声明 (Disclaimer)

本仓库大量内容基于乐聚官方开源生态，请务必阅读以下条款：

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


<a id="hardware-prerequisites"></a>
## 💻 硬件与阅读门槛 (Hardware & Prerequisites)

文档按运行场景、联调范围与部署门槛标注；与 [`kuavo_notes/README.md`](./kuavo_notes/README.md) 保持一致：

| 标记 | 含义 |
| --- | --- |
| 🟢 | **无需 Kuavo 真机**：仿真、离线训练／数据处理、代码阅读或通用参考资料 |
| 🟡 | **需 Kuavo 4 Pro 真机 · 单机侧／模块侧**：以 NUC 或 Orin 一侧的模块部署、感知、音频、接口或运维调试为主 |
| 🔴 | **需 Kuavo 4 Pro 真机 · 双机或全身／部署**：双机协同、全身控制、RL Sim2Real 或机械臂任务集成等；物理风险须按具体流程评估 |
| 🟢 / 🟡、🟢 / 🔴 | **同时涉及仿真／离线与真机阶段**：按章节分阶段阅读，不要求同时运行；真机侧分别沿用 🟡／🔴 的含义 |
| ⚪ | **待整理／占位**：尚无完整流程，不根据标题猜测运行平台 |

**颜色表示运行门槛与联调范围，不表示完成度或成功状态。** 🟢 不表示功能不能迁移真机；🟡／🔴 不表示已通过真机验收。仅有真机部署建议的条目仍明确注明“仅参考”，具体以“状态／验证范围”和正文为准。Orin 与 NUC 都是整机算力，🟡 不是“没有机器人”；组合标记也不代表两端都已验证成功。

* 🟢 **仿真、训练与 Sim2Sim**：如 4.2、6、15.1–15.3、23.2–23.6、23.8，无需机器人本体。
* 🟡 **真机模块侧**：如 4.3、16、17、21.2／21.3、30、32.1、33、34.0，按各篇核对相机、音频与网络依赖。
* 🔴 **双机／全身／部署**：如 4.4、22.1–22.3、15.4、24.1、28、32.2、34，操作前核对急停、防坠与现场监护。
* 🟢 / 🔴 **跨阶段文档**：22.4 包含离线训练与真机试测；23.1、23.7 含仿真流程及真机部署参考，不能将参考命令当作真机验收结果。

**双机架构（真机联调时）：**

| 角色 | 硬件 | 工作空间 |
|------|------|----------|
| 下位机 | Intel NUC | `~/kuavo-ros-opensource` |
| 上位机 | Jetson Orin NX | `~/kuavo_ros_application` |

---


## 📚 核心知识库目录 (Documentation)

完整索引见 [`kuavo_notes/README.md`](./kuavo_notes/README.md)。下列为 `kuavo_notes/` 中 **58 个专题与资源 Markdown 文件**的分类索引（包含待补充条目，不含索引自身）（不含 [`5功能案例/`](./kuavo_notes/5功能案例/案例目录.md) 内官方案例）。

**标记：** 🟢 无需真机（仿真／离线／参考）· 🟡 真机模块侧 · 🔴 真机双机／全身／部署 · 🟢 / 🟡 或 🟢 / 🔴 跨仿真／离线与真机阶段 · ⚪ 待整理。颜色不表示完成度；详见[硬件与阅读门槛](#hardware-prerequisites)。

### 🟢 / ⚪ 概览、入门与待整理入口

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`0.doc.md`](./kuavo_notes/0.doc.md) | 文档地图与阅读路线 |
| 🟢 | [`0.1.example.md`](./kuavo_notes/0.1.example.md) | 开源 / 闭源边界与双机分工 |
| 🟢 | [`1.start.md`](./kuavo_notes/1.start.md) | 仿真 / 实机环境部署 |
| 🟢 | [`2.first_node.md`](./kuavo_notes/2.first_node.md) | 第一个 ROS 节点 |
| 🟢 | [`接口使用文档.md`](./kuavo_notes/接口使用文档.md) | SDK 接口速查 |
| 🟢 | [`999kuavo_resource.md`](./kuavo_notes/999kuavo_resource.md) | 资源链接汇总 |
| ⚪ | [`29decision_tree.md`](./kuavo_notes/29decision_tree.md) | 决策树 |

### 🟢 / 🟡 导航、地图与网络 — 仿真与真机模块侧

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`3.map_navigation.md`](./kuavo_notes/3.map_navigation.md) | 地图、FAST_LIO 与 Docker 挂载踩坑 |
| 🟢 / 🟡 | [`3.1official_navigation.md`](./kuavo_notes/3.1official_navigation.md) | 官方导航案例集成 |
| 🟡 | [`16.Internet.md`](./kuavo_notes/16.Internet.md) | 上下位机网络配置 |
| 🟡 | [`33.height_map.md`](./kuavo_notes/33.height_map.md) | Livox + elevation_mapping 高程图 |

### 🟢 / 🟡 / 🔴 视觉、抓取与 MoveIt

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`4.1.visual_grasping_route.md`](./kuavo_notes/4.1.visual_grasping_route.md) | 视觉抓取路线概览 |
| 🟢 | [`4.2yolov8_sim.md`](./kuavo_notes/4.2yolov8_sim.md) | 仿真侧 YOLOv8 |
| 🟡 | [`4.3.real_robot_yolo_environment.md`](./kuavo_notes/4.3.real_robot_yolo_environment.md) | 真机 YOLO 环境 |
| 🔴 | [`4.4real_visual_grasp.md`](./kuavo_notes/4.4real_visual_grasp.md) | TF2 视觉抓取 |
| 🟢 | [`6.visual_grasp.md`](./kuavo_notes/6.visual_grasp.md) | 视觉抓取基础 |
| ⚪ | [`9.IK.md`](./kuavo_notes/9.IK.md) | 逆运动学 |
| 🟡 | [`20.gripper_issue.md`](./kuavo_notes/20.gripper_issue.md) | 夹爪安全 |
| 🔴 | [`28.moveit_grasping.md`](./kuavo_notes/28.moveit_grasping.md) | MoveIt 经典版 / OctoMap 双轨抓取 |
| 🟡 | [`34.0.YOLO_HSV_Blue_Cap_Detection.md`](./kuavo_notes/34.0.YOLO_HSV_Blue_Cap_Detection.md) | YOLO + HSV 蓝色瓶盖检测，为双臂旋拧提供目标坐标 |
| 🔴 | [`34.two_arm_coordination.md`](./kuavo_notes/34.two_arm_coordination.md) | 双臂协同：抓瓶 + 拧盖（MoveIt） |

### 🟢 / 🟡 / 🔴 全身、手臂与标定 — 仿真与真机

| 标记 | 文档 | 说明 |
|------|------|------|
| 🔴 | [`10.Tai_Ji.md`](./kuavo_notes/10.Tai_Ji.md) | 太极全身动作案例 |
| 🟢 | [`13.arm_move.md`](./kuavo_notes/13.arm_move.md) | 手臂运动学 / 编舞 |
| ⚪ | [`14.robot_dance.md`](./kuavo_notes/14.robot_dance.md) | 机器人舞蹈 |
| 🟢 / 🔴 | [`18.teaching_gravity_compensation.md`](./kuavo_notes/18.teaching_gravity_compensation.md) | 示教 / 重力补偿 |
| 🔴 | [`26.joint_calibration.md`](./kuavo_notes/26.joint_calibration.md) | 关节标定 |
| 🟢 / 🟡 | [`27.camera_mtion_capture.md`](./kuavo_notes/27.camera_mtion_capture.md) | 相机 / 动捕（仿真可跑；真机需 Orin 侧） |
| 🟢 | [`5.up_down_stair.md`](./kuavo_notes/5.up_down_stair.md) | 上下楼梯（仿真已测，真机未测） |

### 🟡 / 🔴 真机 — 大模型、VLA、语音与人脸

| 标记 | 文档 | 说明 |
|------|------|------|
| ⚪ | [`21.1.connected_AI_large_model.md`](./kuavo_notes/21.1.connected_AI_large_model.md) | 联网大模型语音 |
| 🟡 | [`21.2.local_AI_large_model.md`](./kuavo_notes/21.2.local_AI_large_model.md) | 离线 / 局域网大模型 |
| 🟡 | [`21.3.gemini_model.md`](./kuavo_notes/21.3.gemini_model.md) | Gemini 全双工 |
| 🔴 | [`22.1VLA_grasping.md`](./kuavo_notes/22.1VLA_grasping.md) | VLA 语音抓取（双机 9 终端） |
| 🔴 | [`22.2.tree_VLA_grasp.md`](./kuavo_notes/22.2.tree_VLA_grasp.md) | 行为树版 VLA（`py_trees`） |
| 🔴 | [`22.3.MCP_VLA_grasp.md`](./kuavo_notes/22.3.MCP_VLA_grasp.md) | MCP / VLA 抓取 |
| 🟢 / 🔴 | [`22.4.Lerobot_grasp.md`](./kuavo_notes/22.4.Lerobot_grasp.md) | ACT 数据、训练、NUC 部署与真机试测；精抓未达标，子线归档 |
| 🔴 | [`24.1.visual_tracking.md`](./kuavo_notes/24.1.visual_tracking.md) | 头身协同视觉跟随 |
| 🟡 | [`30.AI_image_identification.md`](./kuavo_notes/30.AI_image_identification.md) | VLM 图像触发 |
| 🟡 | [`32.1.face_recognition.md`](./kuavo_notes/32.1.face_recognition.md) | 人脸识别 |
| 🔴 | [`32.2.face_recognition_traking.md`](./kuavo_notes/32.2.face_recognition_traking.md) | 人脸跟踪融合 |

### 🟢 → 🔴 强化学习 · 行走与 Gym

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`7.1.gym_RL.md`](./kuavo_notes/7.1.gym_RL.md) | Gym 版 RL 环境 |
| 🟢 | [`7.2.gym_RL_doc.md`](./kuavo_notes/7.2.gym_RL_doc.md) | Gym 版 RL 代码导读 |
| 🟢 | [`8.imitation_learning.md`](./kuavo_notes/8.imitation_learning.md) | 模仿学习 / Lerobot |
| 🟢 | [`15.1.RL_lab_train.md`](./kuavo_notes/15.1.RL_lab_train.md) | Isaac Lab 行走训练 |
| 🟢 | [`15.2RL_lab_analysis_code.md`](./kuavo_notes/15.2RL_lab_analysis_code.md) | 奖励 / 域随机化拆解 |
| 🟢 | [`15.3RL_lab_sim_to_sim.md`](./kuavo_notes/15.3RL_lab_sim_to_sim.md) | MuJoCo Sim2Sim |
| 🔴 | [`15.4RL_lab_sim_to_real.md`](./kuavo_notes/15.4RL_lab_sim_to_real.md) | 行走 Sim2Real 真机 |

### 🟢 / 🔴 强化学习 · S49 动作跟踪与部署参考

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 / 🔴 | [`23.1.RL_dance_terminal_commands.md`](./kuavo_notes/23.1.RL_dance_terminal_commands.md) | 舞蹈终端命令全集 |
| 🟢 | [`23.2.RL_dance_overview.md`](./kuavo_notes/23.2.RL_dance_overview.md) | S49 舞蹈 RL 总览与当前状态 |
| 🟢 | [`23.3.RL_dance_motion_data.md`](./kuavo_notes/23.3.RL_dance_motion_data.md) | 舞蹈 CSV / v18 arms-only 参考数据 |
| 🟢 | [`23.4.RL_dance_train.md`](./kuavo_notes/23.4.RL_dance_train.md) | S49 训练（当前 115 维 obs） |
| 🟢 | [`23.5.RL_dance_reward_iterate.md`](./kuavo_notes/23.5.RL_dance_reward_iterate.md) | 奖励 v1–v18、视频/TensorBoard 调参与 RUN_CONFIG |
| 🟢 | [`23.6.RL_dance_sim2sim.md`](./kuavo_notes/23.6.RL_dance_sim2sim.md) | MuJoCo 舞蹈 Sim2Sim 与观测顺序 |
| 🟢 / 🔴 | [`23.7.RL_dance_deploy_hybrid.md`](./kuavo_notes/23.7.RL_dance_deploy_hybrid.md) | duck_sit 与 Hybrid WBC+RL 部署 |
| 🟢 | [`23.8.RL_dance_pure_rl_sim2sim_debug.md`](./kuavo_notes/23.8.RL_dance_pure_rl_sim2sim_debug.md) | 纯 RL 对齐、WBC seed 接管与 8 处 Bug |

### 🟢 世界模型

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`31.1.world_model.md`](./kuavo_notes/31.1.world_model.md) | TD-MPC2 / `leju_robot_wm` 实验 |

### 🟢 / 🟡 / 🔴 运维、遥控与排障

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟡 | [`17.h12_remote_control.md`](./kuavo_notes/17.h12_remote_control.md) | H12 遥控器 |
| 🟢 / 🔴 | [`19.tremble_rosbag.md`](./kuavo_notes/19.tremble_rosbag.md) | 抖动 rosbag 排障 |
| 🟡 | [`25.update.md`](./kuavo_notes/25.update.md) | 官方包升级与 launch 排障 |

辅助脚本：[`scripts/analyze_r_takeover_bag.py`](./kuavo_notes/scripts/analyze_r_takeover_bag.py)（RL bag 分析，见 23.7–23.8）。


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


## 🚧 训练相关工作空间（快速迭代中）

| 目录 | 状态 | 说明 |
|------|------|------|
| [`leju_robot_rl/`](./leju_robot_rl/) | 链路已通，效果迭代中 | Isaac Lab + PPO：S42 行走、S49 舞蹈 mimic；奖励 / 域随机化 / CSV 动作仍在调 |
| [`leju_robot_wm/`](./leju_robot_wm/) | 链路已通，效果迭代中 | TD-MPC2 世界模型：舞蹈 / 速度任务；仿真表现尚未稳定到可部署 |
| [`kuavo-rl-opensource/`](./kuavo-rl-opensource/) | 部署侧随 RL 迭代 | MuJoCo / Docker / 真机 deploy；ONNX 与 `humanoidController` 需与训练侧对齐 |

**给克隆者：** 本仓库提供的是**可复现的流水线与踩坑记录**，不是「开箱即用的最终策略」。欢迎 fork 后修改 `mdp/rewards.py`、任务配置、根目录 `kuavo_action_*.csv` 等继续实验。细节见各目录 README。

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
├── automove_and_link.sh      # 作者本机目录维护脚本（克隆者可忽略）
└── SYMLINK_LAYOUT.md         # 软链接说明（克隆者可忽略）
```

各目录说明见对应 `README.md`。

**说明：** 真机源码主要在 NUC / Orin 上维护；本仓库为开发机上的**笔记与代码归档**。魔改影子目录与 RL 子仓**仍在迭代**，可能与文档不同步。真机不会随 `git push` 自动变更。

---


## 🛠️ 技术栈

本仓库在 Kuavo 4 Pro 上覆盖**双机 ROS 控制 → 仿真/RL 训练 → 视觉与 VLA 交互 → ONNX 真机部署**全链路。真机 NUC（`192.168.26.1`）跑下位机，Orin NX（`192.168.26.12`）跑感知与大模型，离线训练在带 NVIDIA GPU 的 Ubuntu 主机完成。

| 领域 | 关键技术 | 本仓库目录 |
|------|----------|------------|
| 基础平台 | Ubuntu 20.04/22.04 · Docker（mpc_wbc **v1.3.0**） · Conda · catkin · `ROBOT_VERSION` 42–49 | 全仓 |
| 开发语言 | **C++17**（`roscpp`、OCS2/WBC、MoveIt、ONNX 推理节点） · **Python 3**（`rospy`、视觉/AI 脚本、LeRobot、Flask 微服务） | 全仓 |
| 机器人控制 | ROS 1 Noetic · OCS2/MPC · WBC · EtherCAT · `humanoid_controllers` | [`kuavo-ros-opensource/`](./kuavo-ros-opensource/) |
| 运动规划 | `motion_capture_ik` · MoveIt/TRAC-IK/OMPL · Pinocchio · py_trees | [`kuavo-ros-opensource/`](./kuavo-ros-opensource/) |
| 仿真 | Gazebo · MuJoCo **3.0.1** · Isaac Sim **4.2.0** | [`kuavo-ros-opensource/`](./kuavo-ros-opensource/) · [`kuavo-rl-opensource/`](./kuavo-rl-opensource/) |
| 强化学习 | Isaac Lab **1.4.1** · rsl_rl/PPO · Isaac Gym（遗留） · ONNX Runtime 50Hz | [`leju_robot_rl/`](./leju_robot_rl/) · [`kuavo-rl-opensource/`](./kuavo-rl-opensource/) |
| 世界模型 | TD-MPC2 | [`leju_robot_wm/`](./leju_robot_wm/) |
| 视觉感知 | YOLOv8 · OpenCV/TF2 · InsightFace · Orbbec/RealSense · AprilTag | [`kuavo_ros_application/`](./kuavo_ros_application/) |
| 抓取操作 | IK 服务 / MoveIt+OctoMap · LejuClaw · LeRobot ACT | 两机协同 |
| 导航建图 | FAST-LIO · Livox · elevation_mapping · 官方导航栈 | [`kuavo-ros-opensource/`](./kuavo-ros-opensource/) |
| 大模型/VLA | Ollama/Qwen2 · Gemini Live · Faster-Whisper/VITS · MCP | [`kuavo_ros_application/`](./kuavo_ros_application/) |
| 数据调试 | rosbag · LET 数据集 · URDF/USD · Foxglove · 龙门架/急停 | [`kuavo_notes/`](./kuavo_notes/) |

**三条主链路：**
- **行走 RL：** Isaac Lab 训练 → 导出 `.onnx` → MuJoCo 验证 → NUC 真机加载
- **视觉抓取：** YOLO 检测 → TF2 转 3D → IK/MoveIt → LejuClaw 闭合
- **语音 VLA：** ASR/LLM 解析意图 → YOLO 定位 → IK 抓取（双机多终端）

> 闭源含 `hardware_plant`、`humanoid_wbc` 等 `.so`；踩坑实录与版本号细节见 [`kuavo_notes/README.md`](./kuavo_notes/README.md)。


---


## 📤 推送 GitHub 建议

* **本地体积（2026-06 整理后）：** 本仓库工作文件（不含 `.git`）约 **~1.5GB**；含各子仓 `.git` 历史约 **~3.5GB**。训练 log / video（约 **17GB**）已外置至 `~/kuavo_all/_training_logs`，**不在本目录内**。早期「全仓 ~26GB」说法已过时，**请勿整包 push**。
* **GitHub 推送建议（体积控制）：** 优先 `kuavo_notes/` + `kuavo-ros-opensource/` + `kuavo_ros_application/` + 根 README（约 **100MB 级**）；RL / WM / 部署子仓按需精选，勿含 checkpoint。
* 推送 RL/WM/部署整仓时，务必 `.gitignore` 排除 `logs/`、`outputs/`、`build/`、`devel/`、`videos/`、`*.pt`、`*.onnx` 等大文件。

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
