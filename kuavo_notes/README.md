# kuavo_notes — 实战文档索引

Kuavo 4 Pro 二次开发的**过程记录**：环境部署、实机踩坑、终端命令、完整源码归档。  
与 [`../kuavo-ros-opensource`](../kuavo-ros-opensource/)、[`../kuavo_ros_application`](../kuavo_ros_application/) 中的魔改代码配套阅读。

**标记：** 🟢 无需真机 · 🟡 需真机（单机侧）· 🔴 需真机（双机/全身/部署）。与 [`../README.md`](../README.md)「硬件与阅读门槛」一致。

---

## 📋 全部文档（52 篇，按主题分类）

不含 [`5功能案例/案例目录.md`](./5功能案例/案例目录.md) 内官方改写案例；官方案例见文末。

**标记：** 🟢 无需真机 · 🟡 需真机（单机侧）· 🔴 需真机（双机/全身/部署）。与 [`../README.md`](../README.md)「硬件与阅读门槛」一致。

### 🟢 概览与入门

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`0.doc.md`](./0.doc.md) | 文档地图与阅读路线 |
| 🟢 | [`0.1.example.md`](./0.1.example.md) | 开源 / 闭源边界与双机分工 |
| 🟢 | [`1.start.md`](./1.start.md) | 仿真 / 实机环境部署 |
| 🟢 | [`2.first_node.md`](./2.first_node.md) | 第一个 ROS 节点 |
| 🟢 | [`接口使用文档.md`](./接口使用文档.md) | SDK 接口速查 |
| 🟢 | [`999kuavo_resource.md`](./999kuavo_resource.md) | 资源链接汇总 |
| 🟢 | [`29decision_tree.md`](./29decision_tree.md) | 决策树 |

### 🟡 真机 · 单机侧 — 导航、地图与网络

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟡 | [`3.map_navigation.md`](./3.map_navigation.md) | 地图、FAST_LIO 与 Docker 挂载踩坑 |
| 🟡 | [`3.1official_navigation.md`](./3.1official_navigation.md) | 官方导航案例集成 |
| 🟡 | [`16.Internet.md`](./16.Internet.md) | 上下位机网络配置 |

### 🟢 / 🟡 / 🔴 视觉、抓取与 MoveIt

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`4.1.visual_grasping_route.md`](./4.1.visual_grasping_route.md) | 视觉抓取路线概览 |
| 🟢 | [`4.2yolov8_sim.md`](./4.2yolov8_sim.md) | 仿真侧 YOLOv8 |
| 🟡 | [`4.3.real_robot_yolo_environment.md`](./4.3.real_robot_yolo_environment.md) | 真机 YOLO 环境 |
| 🔴 | [`4.4real_visual_grasp.md`](./4.4real_visual_grasp.md) | TF2 视觉抓取 |
| 🔴 | [`6.visual_grasp.md`](./6.visual_grasp.md) | 视觉抓取基础 |
| 🟡 | [`9.IK.md`](./9.IK.md) | 逆运动学 |
| 🟡 | [`20.gripper_issue.md`](./20.gripper_issue.md) | 夹爪安全 |
| 🔴 | [`28.moveit_grasping.md`](./28.moveit_grasping.md) | MoveIt 经典版 / OctoMap 双轨抓取 |

### 🟡 / 🔴 真机 — 全身、手臂与标定

| 标记 | 文档 | 说明 |
|------|------|------|
| 🔴 | [`10.Tai_Ji.md`](./10.Tai_Ji.md) | 太极全身动作案例 |
| 🟡 | [`13.arm_move.md`](./13.arm_move.md) | 手臂运动学 / 编舞 |
| 🔴 | [`14.robot_dance.md`](./14.robot_dance.md) | 机器人舞蹈 |
| 🔴 | [`18.teaching_gravity_compensation.md`](./18.teaching_gravity_compensation.md) | 示教 / 重力补偿 |
| 🔴 | [`26.joint_calibration.md`](./26.joint_calibration.md) | 关节标定 |
| 🟡 | [`27.camera_mtion_capture.md`](./27.camera_mtion_capture.md) | 相机 / 动捕 |
| 🔴 | [`5.up_down_stair.md`](./5.up_down_stair.md) | 上下楼梯（真机） |

### 🟡 / 🔴 真机 — 大模型、VLA、语音与人脸

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟡 | [`21.1.connected_AI_large_model.md`](./21.1.connected_AI_large_model.md) | 联网大模型语音 |
| 🟡 | [`21.2.local_AI_large_model.md`](./21.2.local_AI_large_model.md) | 离线 / 局域网大模型 |
| 🟡 | [`21.3.gemini_model.md`](./21.3.gemini_model.md) | Gemini 全双工 |
| 🔴 | [`22.1VLA_grasping.md`](./22.1VLA_grasping.md) | VLA 语音抓取（双机 9 终端） |
| 🔴 | [`22.2.tree_VLA_grasp.md`](./22.2.tree_VLA_grasp.md) | 行为树版 VLA（`py_trees`） |
| 🔴 | [`22.3.MCP_LeRobot_VLA_grasp.md`](./22.3.MCP_LeRobot_VLA_grasp.md) | MCP / LeRobot VLA |
| 🔴 | [`24.1.visual_tracking.md`](./24.1.visual_tracking.md) | 头身协同视觉跟随 |
| 🟡 | [`30.AI_image_identification.md`](./30.AI_image_identification.md) | VLM 图像触发 |
| 🟡 | [`32.1.face_recognition.md`](./32.1.face_recognition.md) | 人脸识别 |
| 🟡 | [`32.2.face_recognition_traking.md`](./32.2.face_recognition_traking.md) | 人脸跟踪融合 |

### 🟢 → 🔴 强化学习 · 行走与 Gym

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`7.1.gym_RL.md`](./7.1.gym_RL.md) | Gym 版 RL 环境 |
| 🟢 | [`7.2.gym_RL_doc.md`](./7.2.gym_RL_doc.md) | Gym 版 RL 代码导读 |
| 🟢 | [`8.imitation_learning.md`](./8.imitation_learning.md) | 模仿学习 / Lerobot |
| 🟢 | [`15.1.RL_lab_train.md`](./15.1.RL_lab_train.md) | Isaac Lab 行走训练 |
| 🟢 | [`15.2RL_lab_analysis_code.md`](./15.2RL_lab_analysis_code.md) | 奖励 / 域随机化拆解 |
| 🟢 | [`15.3RL_lab_sim_to_sim.md`](./15.3RL_lab_sim_to_sim.md) | MuJoCo Sim2Sim |
| 🔴 | [`15.4RL_lab_sim_to_real.md`](./15.4RL_lab_sim_to_real.md) | 行走 Sim2Real 真机 |

### 🟢 → 🔴 强化学习 · S49 全身舞

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`23.1.RL_dance_overview.md`](./23.1.RL_dance_overview.md) | S49 舞蹈 RL 总览与分支纪律 |
| 🟢 | [`23.2.RL_dance_motion_data.md`](./23.2.RL_dance_motion_data.md) | 舞蹈 CSV / 动作数据准备 |
| 🟢 | [`23.3.RL_dance_train.md`](./23.3.RL_dance_train.md) | S49 训练（115 维 obs、mimic 奖励） |
| 🟢 | [`23.4.RL_dance_sim2sim.md`](./23.4.RL_dance_sim2sim.md) | MuJoCo 舞蹈验证 |
| 🔴 | [`23.5.RL_dance_deploy_hybrid.md`](./23.5.RL_dance_deploy_hybrid.md) | 舞蹈真机混合部署 |
| 🔴 | [`23.6.RL_dance_terminal_commands.md`](./23.6.RL_dance_terminal_commands.md) | 舞蹈终端命令全集 |

### 🟢 世界模型

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟢 | [`31.1.world_model.md`](./31.1.world_model.md) | TD-MPC2 / `leju_robot_wm` 实验 |

### 🟡 / 🔴 真机 — 运维、遥控与排障

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟡 | [`17.h12_remote_control.md`](./17.h12_remote_control.md) | H12 遥控器 |
| 🔴 | [`19.tremble_rosbag.md`](./19.tremble_rosbag.md) | 抖动 rosbag 排障 |
| 🟡 | [`25.update.md`](./25.update.md) | 官方包升级与 launch 排障 |

辅助脚本：[`scripts/analyze_r_takeover_bag.py`](./scripts/analyze_r_takeover_bag.py)（RL bag 分析，见 23.5）。

### 📎 官方案例（参考）

[`5功能案例/案例目录.md`](./5功能案例/案例目录.md) — 乐聚官方功能案例索引（本仓库未改写）。

---

## 辅助脚本

| 路径 | 说明 |
|------|------|
| [`scripts/analyze_r_takeover_bag.py`](./scripts/analyze_r_takeover_bag.py) | RL bag 分析（23.5） |

---

文档内路径以真机为准：NUC `~/kuavo-ros-opensource`，Orin `~/kuavo_ros_application`。本机可用 `~/kuavo_all` 软链接到 [`kuavo-dev-notes`](../README.md)。
