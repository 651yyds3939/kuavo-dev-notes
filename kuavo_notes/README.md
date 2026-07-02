# kuavo_notes — 实战文档索引

Kuavo 4 Pro 二次开发的**过程记录**：环境部署、实机踩坑、终端命令、完整源码归档。  
与 [`../kuavo-ros-opensource`](../kuavo-ros-opensource/)、[`../kuavo_ros_application`](../kuavo_ros_application/) 中的魔改代码配套阅读。

**标记：** 🟢 无需真机 · 🟡 需真机（单机侧）· 🔴 需真机（双机/全身/部署）。与 [`../README.md`](../README.md)「硬件与阅读门槛」一致。

---

## 🛠️ 技术栈

本目录 **54 篇笔记** 实际用到的技术汇总如下。终端命令、踩坑与源码见各文档正文；接口速查见 [`接口使用文档.md`](./接口使用文档.md)。

**双机分工：** NUC（`192.168.26.1`）跑 ROS Master、OCS2/MPC/WBC、IK/MoveIt、RL ONNX 与 TTS；Orin NX（`192.168.26.12`）跑相机、YOLO、ASR/LLM、Gemini 网关。跨机走 ROS 话题、HTTP `:5000`、UDP `:7000`。

| 领域 | 技术 | 代表笔记 |
|------|------|----------|
| 环境部署 | Ubuntu 20.04/22.04 · Docker mpc_wbc **v1.3.0** / IL **v0.6.1** · catkin C++17 · Conda | [`1.start`](./1.start.md) [`8`](./8.imitation_learning.md) |
| 开发语言 | **C++17**（控制/规划底层） · **Python 3**（ROS 节点、AI、数据采集） · Bash（终端编排） | [`1.start`](./1.start.md) [`2`](./2.first_node.md) |
| 双机与网络 | 静态 IP · `ROS_MASTER_URI` · SSH/SSHFS · PulseAudio 声卡排障 | [`16`](./16.Internet.md) [`0.1`](./0.1.example.md) |
| 全身控制 | ROS 1 Noetic · OCS2/MPC · WBC · EtherCAT · `/cmd_vel` · H12Pro | [`17`](./17.h12_remote_control.md) [`15.4`](./15.4RL_lab_sim_to_real.md) [`32.2`](./32.2.face_recognition_traking.md) |
| 手臂与 IK | `motion_capture_ik` · `humanoid_plan_arm_trajectory` · Pinocchio · 示教/标定 | [`9`](./9.IK.md) [`13`](./13.arm_move.md) [`18`](./18.teaching_gravity_compensation.md) [`26`](./26.joint_calibration.md) |
| 仿真 | Gazebo · MuJoCo **3.0.1** · `humanoid_controllers` 统一 launch | [`1.start`](./1.start.md) [`4.2`](./4.2yolov8_sim.md) [`15.3`](./15.3RL_lab_sim_to_sim.md) |
| 行走强化学习 | Isaac Sim **4.2** · Isaac Lab **1.4.1** · rsl_rl/PPO · 87 维 obs · ONNX 真机 50Hz | [`15.1`](./15.1.RL_lab_train.md)–[`15.4`](./15.4RL_lab_sim_to_real.md) |
| 舞蹈强化学习 | S49 · 115 维 obs · mimic CSV · Sim2Sim → 混合真机部署 | [`23.1`](./23.1.RL_dance_overview.md)–[`23.6`](./23.6.RL_dance_terminal_commands.md) |
| 遗留 Gym RL | Isaac Gym Preview 4 · `kuavo-rl-opensource` beta | [`7.1`](./7.1.gym_RL.md) [`7.2`](./7.2.gym_RL_doc.md) |
| 世界模型 | TD-MPC2 · `leju_robot_wm` | [`31.1`](./31.1.world_model.md) |
| 视觉检测 | YOLOv8 · OpenCV/cv_bridge · TF2 · Orbbec/RealSense | [`4.1`](./4.1.visual_grasping_route.md)–[`4.4`](./4.4real_visual_grasp.md) [`6`](./6.visual_grasp.md) |
| 运动规划抓取 | MoveIt/TRAC-IK/OMPL · OctoMap · 官方 IK 双轨 · LejuClaw | [`28`](./28.moveit_grasping.md) [`34`](./34.two_arm_coordination.md) |
| 视觉跟随/人脸 | InsightFace · YOLO+PID · Gemini 多模态融合 | [`24.1`](./24.1.visual_tracking.md) [`32.1`](./32.1.face_recognition.md) [`32.2`](./32.2.face_recognition_traking.md) |
| 导航与建图 | FAST-LIO · Livox Avia/Mid-360 · elevation_mapping · 官方导航栈 | [`3`](./3.map_navigation.md) [`3.1`](./3.1official_navigation.md) [`33`](./33.height_map.md) |
| 本地大模型语音 | Ollama/Qwen2-7B · Faster-Whisper · VITS · Flask `:5000` | [`21.2`](./21.2.local_AI_large_model.md) |
| 云端大模型 | Gemini Live API (WSS) · Clash/Proxychains4（禁与 cv2 同进程） | [`21.3`](./21.3.gemini_model.md) |
| VLA 与模仿学习 | 语音→LLM→YOLO→抓取 · py_trees · MCP · LeRobot ACT | [`22.1`](./22.1VLA_grasping.md)–[`22.4`](./22.4.Lerobot_grasp.md) |
| 动捕与遥操作 | MediaPipe Pose · Quest VR · 手机 IP Webcam | [`27`](./27.camera_mtion_capture.md) |
| 数据与排障 | rosbag · LET 数据集 · URDF/`kuavo.json` · Foxglove · 升级/抖动/夹爪 | [`8`](./8.imitation_learning.md) [`19`](./19.tremble_rosbag.md) [`20`](./20.gripper_issue.md) [`25`](./25.update.md) [`999`](./999kuavo_resource.md) |

> **开源可改：** Demo、SDK、IK、py_trees、MoveIt/OCS2 配置。**闭源二进制：** `hardware_plant`、`humanoid_wbc`、EtherCAT 主站。见 [`0.1.example.md`](./0.1.example.md)。


---

## 📋 全部文档（54 篇，按主题分类）

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
| 🟡 | [`33.height_map.md`](./33.height_map.md) | Livox + elevation_mapping 高程图 |

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
| 🔴 | [`34.two_arm_coordination.md`](./34.two_arm_coordination.md) | 双臂协同：抓瓶 + 拧盖（MoveIt） |

### 🟡 / 🔴 真机 — 全身、手臂与标定

| 标记 | 文档 | 说明 |
|------|------|------|
| 🔴 | [`10.Tai_Ji.md`](./10.Tai_Ji.md) | 太极全身动作案例 |
| 🟡 | [`13.arm_move.md`](./13.arm_move.md) | 手臂运动学 / 编舞 |
| 🔴 | [`14.robot_dance.md`](./14.robot_dance.md) | 机器人舞蹈 |
| 🔴 | [`18.teaching_gravity_compensation.md`](./18.teaching_gravity_compensation.md) | 示教 / 重力补偿 |
| 🔴 | [`26.joint_calibration.md`](./26.joint_calibration.md) | 关节标定 |
| 🟢 / 🟡 | [`27.camera_mtion_capture.md`](./27.camera_mtion_capture.md) | 相机 / 动捕（仿真可跑；真机需 Orin 侧） |
| 🟢 | [`5.up_down_stair.md`](./5.up_down_stair.md) | 上下楼梯（仿真已测，真机未测） |

### 🟡 / 🔴 真机 — 大模型、VLA、语音与人脸

| 标记 | 文档 | 说明 |
|------|------|------|
| 🟡 | [`21.1.connected_AI_large_model.md`](./21.1.connected_AI_large_model.md) | 联网大模型语音 |
| 🟡 | [`21.2.local_AI_large_model.md`](./21.2.local_AI_large_model.md) | 离线 / 局域网大模型 |
| 🟡 | [`21.3.gemini_model.md`](./21.3.gemini_model.md) | Gemini 全双工 |
| 🔴 | [`22.1VLA_grasping.md`](./22.1VLA_grasping.md) | VLA 语音抓取（双机 9 终端） |
| 🔴 | [`22.2.tree_VLA_grasp.md`](./22.2.tree_VLA_grasp.md) | 行为树版 VLA（`py_trees`） |
| 🔴 | [`22.3.MCP_VLA_grasp.md`](./22.3.MCP_VLA_grasp.md) | MCP / VLA 抓取 |
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
