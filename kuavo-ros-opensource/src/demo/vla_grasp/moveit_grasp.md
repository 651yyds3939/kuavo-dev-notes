# Kuavo 双臂 MoveIt 抓取 — 终端发车手册

> **操作习惯**：在**上位机**（`leju_kuavo@192.168.26.12`）统一开终端；下位机节点通过 `ssh lab@192.168.26.1` 远程拉起。  
> **曲肘时机**：只有**终端 8** 抓取脚本启动后才开始曲肘和抓取，前面全是铺垫。  
> **完整踩坑记录**（含上位机改相机、提示词、每轮 TCP 标定）：[`question.md`](question.md)（**强烈建议通读**）。

发车前 `Ctrl+C` 关掉所有旧窗口，严格按 **终端 1 → 8** 顺序执行。

---

## 环境与网络

| 角色 | 用户 / 主机 | IP | 工作空间 |
|------|-------------|-----|----------|
| 下位机 NUC | `lab@192.168.26.1` | `192.168.26.1` | `~/kuavo-ros-opensource` |
| 上位机 | `leju_kuavo@192.168.26.12` | `192.168.26.12` | `~/kuavo_ros_application` |

```bash
# 两台机器均需（或写入 ~/.bashrc）
export ROS_MASTER_URI=http://kuavo_master:11311
export ROS_IP=<本机主机名或 IP，下位机多为 kuavo_master>
```

---

## 脚本一览（当前维护）

| 脚本 | 终端 3 | 用途 | 是否改 |
|------|--------|------|--------|
| **`moveit_auto_grasp.py`** | `move_group.launch` | **经典版**：MoveIt 做 IK，轨迹≈vla 硬编码，收手肩膀外摆 | ✅ 主开发 |
| **`moveit_octomap_grasp.py`** | `move_group_octomap.launch` | **点云版**：抓取同经典；抬升后 OctoMap+OMPL 收手 | ✅ 主开发 |
| `auto_grasp_TF2.py` | 无 MoveIt | 厂家 IK + 硬编码基准 | ❌ 参考勿改 |
| `vla_auto_grasp_daemon.py` | 无 MoveIt | VLA 守护；收手外摆逻辑来源 | ❌ 参考勿改 |
| `kuavo_state_publisher.py` | — | 终端 2 必开 | ✅ |
| `look_down.py` | — | 终端 4 低头 | ✅ |

---

## 两种发车模式（终端 3 + 8 二选一）

| 模式 | 终端 3 | 终端 8 | 点云 | 收手 |
|------|--------|--------|------|------|
| **经典版**（默认，更稳） | `move_group.launch` | `moveit_auto_grasp.py` | ❌ | vla 肩膀外摆 75° |
| **OctoMap 点云版** | `move_group_octomap.launch` | `moveit_octomap_grasp.py` | ✅ 抬升后 | OMPL+OctoMap（失败→vla） |

终端 **1、2、4、5、6、7** 两种模式相同。**勿混用**两种 move_group。

---

## 快速复制：经典版全流程

```bash
# ── 终端 1 · 下位机（sudo）──
ssh lab@192.168.26.1
sudo su
cd kuavo-ros-opensource && source devel/setup.bash
roslaunch humanoid_controllers load_kuavo_real.launch cali:=true

# ── 终端 2 · 下位机 SSH ──
ssh lab@192.168.26.1
cd kuavo-ros-opensource && source devel/setup.bash
python3 src/demo/vla_grasp/kuavo_state_publisher.py

# ── 终端 3 · 下位机 SSH ──
ssh lab@192.168.26.1
cd kuavo-ros-opensource && source devel/setup.bash
roslaunch kuavo_arm_moveit_config move_group.launch

# ── 终端 4 · 下位机 SSH ──
ssh lab@192.168.26.1
cd kuavo-ros-opensource && source devel/setup.bash
python3 src/demo/vla_grasp/look_down.py

# ── 终端 5 · 上位机本地 ──
cd ~/kuavo_ros_application && source devel/setup.bash
roslaunch dynamic_biped load_robot_head.launch

# ── 终端 6 · 上位机本地 ──
cd ~/kuavo_ros_application && source devel/setup.bash
python3 src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py

# ── 终端 7 · 上位机本地（推荐）──
source /opt/ros/noetic/setup.bash
rqt_image_view

# ── 终端 8 · 下位机 SSH（发车）──
ssh lab@192.168.26.1
cd kuavo-ros-opensource && source devel/setup.bash
python3 src/demo/vla_grasp/moveit_auto_grasp.py
```

---

## 快速复制：OctoMap 点云版（仅替换终端 3、8）

```bash
# 终端 3 改为：
roslaunch kuavo_arm_moveit_config move_group_octomap.launch

# 终端 5 启动后验收（任意终端）：
rostopic hz /camera/depth/color/points   # 须 ~15–22 Hz

# 终端 8 改为：
python3 src/demo/vla_grasp/moveit_octomap_grasp.py
```

**OctoMap 成功日志关键字**：`🗺️ 阶段B`、`MoveIt 碰撞感知规划: [OctoMap外移`。

---

## MoveIt 实际用到哪一步

> **常见误解**：文件名带 MoveIt ≠ 全程 OMPL ≠ 全程点云。

```
经典版 moveit_auto_grasp.py          OctoMap 版 moveit_octomap_grasp.py
────────────────────────────         ────────────────────────────────────
视觉 TF2+YOLO                        同左
曲肘→预瞄→插入→抬升  硬编码关节       同左（阶段 A）
IK                   /compute_ik     同左
点云 / OctoMap       ❌               ✅ 仅抬升后阶段 B 收手
收手                 vla 肩膀外摆      OMPL+点云+桌面盒（失败→vla）
```

| 时机 | 经典版 MoveIt 作用 |
|------|-------------------|
| 启动 | 加载 `biped_s49` 模型（URDF collision WARN 此时出现，可忽略） |
| 抓取前 | `/compute_ik` 算抓握/预瞄/抬升关节角 |
| 执行 | **不用** OMPL；`execute_single_pose` 直发 `/kuavo_arm_target_poses` |
| 收手 | **不用** 点云；`execute_vla_style_return()` |
| 异常 | `_safe_return_both_arms()` 才调 OMPL |

日志若结尾为 `肩膀外摆避障` → 经典版，**全程无点云**。  
日志若有 `阶段B` / `OctoMap外移` → 点云版收手。

---

## 总览

| 终端 | 名称 | 运行位置 | 工作空间 | 跳过？ |
|------|------|----------|----------|--------|
| **1** | 底层上电 | 🤖 下位机 `sudo` | `kuavo-ros-opensource` | ❌ |
| **2** | 关节状态引渡 | 🤖 下位机 SSH | `kuavo-ros-opensource` | ❌ |
| **3** | MoveIt | 🤖 下位机 SSH | `kuavo-ros-opensource` | ❌ |
| **4** | 头部低头 | 🤖 下位机 SSH | `kuavo-ros-opensource` | ❌ |
| **5** | 深度相机 | 🖥️ 上位机本地 | `kuavo_ros_application` | ❌ |
| **6** | YOLO + TF2 | 🖥️ 上位机本地 | `kuavo_ros_application` | ❌ |
| **7** | rqt 画面 | 🖥️ 上位机本地 | 任意 ROS 环境 | 推荐 |
| **8** | 抓取主程序 | 🤖 下位机 SSH | `kuavo-ros-opensource` | ❌ |

**下位机 SSH 前缀**（终端 1～4、8）：

```bash
ssh lab@192.168.26.1
cd kuavo-ros-opensource
source devel/setup.bash
```

---

## 终端 1：底层上电

```bash
ssh lab@192.168.26.1
sudo su
cd kuavo-ros-opensource
source devel/setup.bash
roslaunch humanoid_controllers load_kuavo_real.launch cali:=true
```

**等待**：标定结束、站稳、脖子变硬 → 再开终端 2。

---

## 终端 2：关节状态引渡

```bash
ssh lab@192.168.26.1
cd kuavo-ros-opensource
source devel/setup.bash
python3 src/demo/vla_grasp/kuavo_state_publisher.py
```

**现象**：`👁️ Kuavo 全身 28 轴状态引渡神经元激活`。  
**漏开后果**：MoveIt / 抓取无 `/joint_states`，**必开**。

---

## 终端 3：MoveIt

### 经典版

```bash
ssh lab@192.168.26.1
cd kuavo-ros-opensource
source devel/setup.bash
roslaunch kuavo_arm_moveit_config move_group.launch
```

→ 配对终端 8：`moveit_auto_grasp.py`

### OctoMap 点云版

```bash
roslaunch kuavo_arm_moveit_config move_group_octomap.launch
```

→ 配对终端 8：`moveit_octomap_grasp.py`  
**勿**与 `move_group.launch` 同时运行。

**就绪标志**：`You can start planning now!`

---

## 终端 4：头部低头

```bash
ssh lab@192.168.26.1
cd kuavo-ros-opensource
source devel/setup.bash
python3 src/demo/vla_grasp/look_down.py
```

---

## 终端 5：深度相机（上位机）

```bash
ssh leju_kuavo@192.168.26.12   # 若已在上位机桌面可省略
cd ~/kuavo_ros_application
source devel/setup.bash
roslaunch dynamic_biped load_robot_head.launch
```

### 点云验收（OctoMap 版必做）

上位机 `orbbec_sensor_robot_enable.launch` 用 **depth_image_proc** 生成点云（驱动内 `enable_point_cloud` 已关，防 SIGSEGV）。

```bash
rostopic hz /camera/depth/image_raw
rostopic hz /camera/depth/color/points
rostopic info /camera/depth/color/points   # publisher: vla_depth_to_pointcloud
```

`/vla/yolo_target` 有数据 **≠** 有点云。

### 相机卡死

**现象**：`Received signal: 11`、`process has died exit code 11`、全部话题 `no new messages`。  
**处理**：`Ctrl+C` 重启终端 5；经典版比 OctoMap 版更省相机负载。详见 [`question.md` §6](question.md)。

---

## 终端 6：YOLO + TF2（上位机）

```bash
cd ~/kuavo_ros_application
source devel/setup.bash
python3 src/ros_vision/detection_industrial_yolo/yolo_box_object_detection/scripts/yolo_box_segment_ros_TF2.py
```

**现象**：`🎯 绝对坐标 (抗晃动): X=..., Y=...`  
**发布**：`/vla/yolo_target`

---

## 终端 7：rqt_image_view（上位机，推荐）

```bash
source /opt/ros/noetic/setup.bash
rqt_image_view
```

确认水瓶被稳定框选后再开终端 8。

---

## 终端 8：抓取主程序（下位机 SSH）

### 经典版 `moveit_auto_grasp.py`

```bash
ssh lab@192.168.26.1
cd kuavo-ros-opensource
source devel/setup.bash
python3 src/demo/vla_grasp/moveit_auto_grasp.py
```

**流程**：init → 视觉 10 帧 → IK 算点 → **曲肘护胸** → 预瞄 12cm → **单段水平插入** → 夹爪 → 抬升 22cm → **vla 肩膀外摆收手** → 松爪

### OctoMap 版 `moveit_octomap_grasp.py`

```bash
python3 src/demo/vla_grasp/moveit_octomap_grasp.py
```

**流程**：阶段 A 同经典 → `clear_octomap` → 插入 → 抬升 → **阶段 B 点云 OMPL 收手**

---

## 发车前检查

```bash
# 下位机
rostopic hz /joint_states
rostopic echo /vla/yolo_target -n 1
rosservice list | grep compute_ik

# OctoMap 版额外
rostopic hz /camera/depth/color/points
```

**禁止同时运行**：`kuavo_moveit_bridge.py`（与终端 8 抢 `/kuavo_arm_target_poses`）。

---

## 关键配置与 TCP 标定

文件：`src/demo/vla_grasp/moveit_auto_grasp.py`（OctoMap 版复用）

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `TCP_OFFSET_X_LEFT` | `-0.018` | 左手 X（负=往后） |
| `TCP_OFFSET_X_RIGHT` | `-0.024` | 右手 X（略多于左手） |
| `TCP_OFFSET_Y_LEFT` | `0.020` | 左手 Y（偏左则减小） |
| `TCP_OFFSET_Y_RIGHT` | `0.065` | 右手 Y（偏右则增大） |
| `SAFE_LOCKED_Z` | `0.385` | 抓握高度 |
| `LIFT_HEIGHT` | `0.22` | 抬升 22cm |

OctoMap 配置：`src/kuavo_arm_moveit_config/config/sensors_3d_octomap.yaml`

---

## 上位机窗口布局

```
┌─────────────────┬─────────────────┐
│  终端 5 相机     │  终端 6 YOLO    │
├─────────────────┼─────────────────┤
│  终端 7 rqt     │  SSH → 终端 8   │
└─────────────────┴─────────────────┘
另开 SSH：终端 1～4 下位机后台常驻
```

---

## 常见问题（简表）

| 问题 | 处理 |
|------|------|
| 无 `/joint_states` | 开终端 2 |
| IK 不可用 | 开终端 3，等 `You can start planning now!` |
| `compute_ik` WARN 后仍成功 | 正常；优先 `/compute_ik` |
| 记得有点云收手但没有 | 你跑的是经典版；换 OctoMap 终端 3+8 |
| 相机 SIGSEGV | 重启终端 5 |
| 胳膊不动 | 勿开 `kuavo_moveit_bridge.py` |
| URDF collision WARN | 启动一次，可忽略 |

详细根因见 [`question.md`](question.md)。

---

## 双 Cursor / 双机改代码指南

| 改什么 | 在哪改 | 怎么打开 |
|--------|--------|----------|
| 抓取脚本、MoveIt launch、TCP | 下位机 | 本仓库 Cursor 或 `ssh lab@192.168.26.1` |
| 相机 launch、Orbbec、点云 | 上位机 | Cursor Remote SSH → `leju_kuavo@192.168.26.12` → `~/kuavo_ros_application` |
| 下位机远程看上位机 | 下位机终端 | `ssh leju_kuavo@192.168.26.12`（密码见运维记录） |
| 挂载上位机目录 | 下位机 | `sshfs leju_kuavo@192.168.26.12:~/kuavo_ros_application ~/remote_upper_ws` |

**为何上位机要单独改相机？** 点云 launch 在 `kuavo_ros_application`，下位机 Cursor 默认看不到。详见 [`question.md` §13](question.md)。

---

## 上位机相机改动摘要（终端 5 相关）

> 完整改前/改后 XML、提示词原文、SIGSEGV 分析见 [`question.md` §13–§16](question.md)。

### 问题

- `/camera/depth/color/points` 在 `rostopic list` 里存在，但 `hz` 为 `no new messages`  
- 第一版开 `enable_point_cloud` + relay 后点云可用，但跑 1～2 轮抓取相机 **SIGSEGV 卡死**  

### 最终稳定方案（`orbbec_sensor_robot_enable.launch`）

| 项 | 值 |
|----|-----|
| `enable_point_cloud` | **`false`**（关驱动内点云） |
| `color_fps` | **30**（原 60） |
| 点云生成 | **`depth_image_proc/point_cloud_xyz`** nodelet |
| 输出话题 | `/camera/depth/color/points` |
| publisher 名 | `vla_depth_to_pointcloud` |

**入口不变**：`roslaunch dynamic_biped load_robot_head.launch`

### 发给上位机 Cursor 的任务摘要

1. 沿 launch 链查 `enable_point_cloud` 默认值  
2. 让 `/camera/depth/color/points` 有 ≥15Hz 数据  
3. 对齐下位机 `sensors_3d_octomap.yaml`  
4. **不要改**下位机仓库与 YOLO  

提示词全文：[`question.md` 附录 C / §13.3](question.md)

---

## 下位机新增/修改文件索引

| 路径 | 说明 |
|------|------|
| `src/demo/vla_grasp/moveit_auto_grasp.py` | 经典版主程序 |
| `src/demo/vla_grasp/moveit_octomap_grasp.py` | 点云版主程序 |
| `src/demo/vla_grasp/launch/depth_to_pointcloud.launch` | 下位机备用点云（通常用上位机内置） |
| `src/kuavo_arm_moveit_config/launch/move_group_octomap.launch` | OctoMap 版 move_group |
| `src/kuavo_arm_moveit_config/launch/sensor_manager_octomap.launch.xml` | OctoMap 传感器管理 |
| `src/kuavo_arm_moveit_config/config/sensors_3d_octomap.yaml` | 点云话题配置 |
| `src/kuavo_arm_moveit_config/launch/biped_s49_moveit_sensor_manager.launch.xml` | 补 sensor manager param |
| `src/kuavo_arm_moveit_config/config/sensors_3d.yaml` | **保持空** `sensors: []` |

---

## 历史参考脚本（禁止修改）

| 文件 | 用途 | 与 MoveIt 版关系 |
|------|------|------------------|
| `auto_grasp_TF2.py` | 厂家 IK + 硬编码；实机验证基准 | init/视觉/曲肘/插入逻辑来源 |
| `vla_auto_grasp_daemon.py` | VLA 守护进程 | **收手肩膀外摆 75°** 来源 |
| `kuavo_moveit_bridge.py` | MoveIt↔真机桥 | **禁止与终端 8 同开** |
| `auto_grasp.py` | 更早版本 | 仅历史参考 |

---

## 文档维护说明

- **`moveit_grasp.md`（本文件）**：终端命令、双模式发车、快速复制、当前参数  
- **`question.md`**：全对话时间线、每一个踩坑、上位机改动、提示词、版本演进、40 条速查表  

若代码再次迭代，请**同时更新**两文件。

---

8 终端就绪后，在**终端 8** 运行抓取脚本即可（经典版或 OctoMap 版二选一）。
