# kuavo_notes — 实战文档索引

Kuavo 4 Pro 二次开发的**过程记录**：环境部署、实机踩坑、终端命令、完整源码归档。  
与 [`../kuavo-ros-opensource`](../kuavo-ros-opensource/)、[`../kuavo_ros_application`](../kuavo_ros_application/) 中的魔改代码配套阅读。

---

## 推荐阅读顺序

### 入门

| 文档 | 内容 |
|------|------|
| [`1.start.md`](./1.start.md) | 仿真 / 实机环境 |
| [`2.first_node.md`](./2.first_node.md) | 第一个 ROS 节点 |
| [`接口使用文档.md`](./接口使用文档.md) | SDK 接口速查 |

### 感知与抓取

| 文档 | 内容 |
|------|------|
| [`4.3.real_robot_yolo_environment.md`](./4.3.real_robot_yolo_environment.md) | 真机 YOLO 环境 |
| [`4.4real_visual_grasp.md`](./4.4real_visual_grasp.md) | TF2 视觉抓取 |
| [`28.moveit_grasping.md`](./28.moveit_grasping.md) | MoveIt 双轨抓取 |
| [`22.1VLA_grasping.md`](./22.1VLA_grasping.md) | VLA 语音抓取 |
| [`22.2.tree_VLA_grasp.md`](./22.2.tree_VLA_grasp.md) | 行为树版 VLA |
| [`22.3.MCP_LeRobot_VLA_grasp.md`](./22.3.MCP_LeRobot_VLA_grasp.md) | MCP / LeRobot |

### 大模型与人脸

| 文档 | 内容 |
|------|------|
| [`21.2.local_AI_large_model.md`](./21.2.local_AI_large_model.md) | 离线 / 局域网大模型 |
| [`21.3.gemini_model.md`](./21.3.gemini_model.md) | Gemini 全双工 |
| [`30.AI_image_identification.md`](./30.AI_image_identification.md) | VLM 图像触发 |
| [`32.1.face_recognition.md`](./32.1.face_recognition.md) | 人脸识别 |
| [`32.2.face_recognition_traking.md`](./32.2.face_recognition_traking.md) | 人脸跟踪融合 |
| [`24.1.visual_tracking.md`](./24.1.visual_tracking.md) | 视觉跟随 |

### 强化学习

| 文档 | 内容 |
|------|------|
| [`15.1`](./15.1.RL_lab_train.md) – [`15.4`](./15.4RL_lab_sim_to_real.md) | Lab 行走 |
| [`23.1`](./23.1.RL_dance_overview.md) – [`23.6`](./23.6.RL_dance_terminal_commands.md) | S49 全身舞 |
| [`7.1.gym_RL.md`](./7.1.gym_RL.md) · [`7.2.gym_RL_doc.md`](./7.2.gym_RL_doc.md) | Gym 版 |

### 世界模型

| 文档 | 内容 |
|------|------|
| [`31.1.world_model.md`](./31.1.world_model.md) | TD-MPC2 / `leju_robot_wm` |

### 其他

| 文档 | 内容 |
|------|------|
| [`3.map_navigation.md`](./3.map_navigation.md) | 地图与 FAST_LIO |
| [`25.update.md`](./25.update.md) | 官方包升级 |
| [`20.gripper_issue.md`](./20.gripper_issue.md) | 夹爪安全 |
| [`5功能案例/案例目录.md`](./5功能案例/案例目录.md) | 官方案例 |

---

## 辅助脚本

| 路径 | 说明 |
|------|------|
| [`scripts/analyze_r_takeover_bag.py`](./scripts/analyze_r_takeover_bag.py) | RL bag 分析（23.5） |

---

文档内路径以真机为准：NUC `~/kuavo-ros-opensource`，Orin `~/kuavo_ros_application`。本机可用 `~/kuavo_all` 软链接到 [`kuavo-dev-notes`](../README.md)。
