# MoveIt 水瓶抓取 — 问题与解决全记录（完整版）

> **本文档**：按对话时间线记录 Kuavo 14-DOF 双臂 MoveIt 抓取开发中的**每一个**问题、根因、解决方法与代码变更。  
> **发车命令**：[`moveit_grasp.md`](moveit_grasp.md)  
> **参考脚本（勿改）**：`auto_grasp_TF2.py`、`vla_auto_grasp_daemon.py`  
> **最后更新**：2026-06-17

---

## 目录

1. [项目背景与硬性约束](#1-项目背景与硬性约束)  
2. [第一阶段：初始需求 — 手搓笛卡尔 IK 引擎](#2-第一阶段初始需求--手搓笛卡尔-ik-引擎)  
3. [第二阶段：文档重写与操作习惯](#3-第二阶段文档重写与操作习惯)  
4. [第三阶段：五代程序 Bug 审查（过快/不准/障碍）](#4-第三阶段五代程序-bug-审查过快不准障碍)  
5. [第四阶段：网络拓扑与上下位机角色](#5-第四阶段网络拓扑与上下位机角色)  
6. [第五阶段：终端 8 启动失败与通信](#6-第五阶段终端-8-启动失败与通信)  
7. [第六阶段：双臂安全与失败残留姿态](#7-第六阶段双臂安全与失败残留姿态)  
8. [第七阶段：路径几何 — 斜插/先上后前碰瓶](#8-第七阶段路径几何--斜插先上后前碰瓶)  
9. [第八阶段：MoveIt 价值质疑与 IK 全面无解](#9-第八阶段moveit-价值质疑与-ik-全面无解)  
10. [第九阶段：实机微调 — TCP / 收手碰桌 / 非活动臂误动](#10-第九阶段实机微调--tcp--收手碰桌--非活动臂误动)  
11. [第十阶段：OctoMap 可行性讨论与实现](#11-第十阶段octomap-可行性讨论与实现)  
12. [第十一阶段：点云话题存在但无数据](#12-第十一阶段点云话题存在但无数据)  
13. [第十二阶段：上位机 Cursor 改相机（提示词 + 具体改动）](#13-第十二阶段上位机-cursor-改相机提示词--具体改动)  
14. [第十三阶段：OctoMap 版实机调试](#14-第十三阶段octomap-版实机调试)  
15. [第十四阶段：TCP 多轮标定与 IK WARN](#15-第十四阶段tcp-多轮标定与-ik-warn)  
16. [第十五阶段：Orbbec 相机 SIGSEGV 卡死](#16-第十五阶段orbbec-相机-sigsegv-卡死)  
17. [第十六阶段：「MoveIt 真在用吗？」与启动 WARN](#17-第十六阶段moveit-真在用吗与启动-warn)  
18. [第十七阶段：「记得有点云收手」记忆混淆](#18-第十七阶段记得有点云收手记忆混淆)  
19. [附录 A：下位机全部文件变更清单](#附录-a下位机全部文件变更清单)  
20. [附录 B：上位机 launch 改前 / 改后](#附录-b上位机-launch-改前--改后)  
21. [附录 C：发给上位机 Cursor 的完整提示词](#附录-c发给上位机-cursor-的完整提示词)  
22. [附录 D：标准 MoveIt Pick 与水瓶伪障碍](#附录-d标准-moveit-pick-与水瓶伪障碍)  
23. [附录 E：问题—原因—解决 速查表（全）](#附录-e问题原因解决-速查表全)  
24. [附录 F：版本演进与当前稳定流程](#附录-f版本演进与当前稳定流程)  
25. [附录 G：关键函数与参数索引](#附录-g关键函数与参数索引)

---

## 1. 项目背景与硬性约束

### 1.1 双机架构（最终确认）

| 角色 | 用户 | IP | 网口 | 工作空间 |
|------|------|-----|------|----------|
| **下位机** NUC | `lab` | `192.168.26.1` | `enx00e04c68345c` | `~/kuavo-ros-opensource` |
| **上位机** 感知工控 | `leju_kuavo` | `192.168.26.12` | 有线连下位机 | `~/kuavo_ros_application` |

- `ROS_MASTER_URI=http://kuavo_master:11311`（下位机 hostname 常解析为此）
- 文档视角：**人在上位机开终端**，终端 5～7 本地；终端 1～4、8 通过 `ssh lab@192.168.26.1` 进下位机
- 用户也曾坐在 **下位机 NUC** 上操作（`lab@NUC11TNKi7`），此时上位机命令需 `ssh leju_kuavo@192.168.26.12`

### 1.2 开发目标（原始需求）

在**不用**厂家 IK `/ik/two_arm_hand_pose_cmd_srv` 的前提下，用 MoveIt 完成：

1. TF2 视觉锁点  
2. 预瞄（肩→瓶退 12cm）  
3. **12cm 水平直线切入**（姿态锁定，仅 xyz 平移）  
4. 夹爪闭合 → 垂直抬升 → 安全收手  

### 1.3 系统级禁令

| 禁令 | 原因 | 违反后果 |
|------|------|----------|
| Python `move_group.compute_cartesian_path()` | Boost.Python 签名 bug | 进程闪退 `ArgumentError` |
| 调用 `/ik/two_arm_hand_pose_cmd_srv` | 需求要求纯 MoveIt | 违背架构 |
| 与 `kuavo_moveit_bridge.py` 同时运行 | 同抢 `/kuavo_arm_target_poses` | 胳膊不动 / 乱动 |
| 两种 `move_group` 同时开 | 重复节点 / 话题冲突 | 不可预测 |

**允许**：ROS 服务 `/compute_ik`、`/compute_cartesian_path`（C++ 服务，非 Python 绑定）

### 1.4 当前脚本矩阵

| 脚本 | MoveIt | 点云 | 收手 | 状态 |
|------|--------|------|------|------|
| `auto_grasp_TF2.py` | 厂家 IK | ❌ | 硬编码 | 参考勿改 |
| `vla_auto_grasp_daemon.py` | 无 | ❌ | 肩膀外摆 75° | 参考勿改 |
| **`moveit_auto_grasp.py`** | **仅 IK** | ❌ | vla 外摆 | **经典版·主用** |
| **`moveit_octomap_grasp.py`** | IK+OMPL | ✅ 收手 | OctoMap→vla | **点云版·主用** |

---

## 2. 第一阶段：初始需求 — 手搓笛卡尔 IK 引擎

### 2.1 用户需求原文摘要

- 预瞄点后执行 12cm 笛卡尔直线切入，四元数全程锁定  
- 用 numpy 切 20～30 个 Pose，循环调 `GetPositionIK`  
- 7 DOF 冗余臂：上一解作 seed，防肘部甩尾（单关节跳变 >30° 丢弃）  
- 封装 `JointTrajectory`，流式发 `/kuavo_arm_target_poses`  

### 2.2 首次实现

**改动文件**：`moveit_auto_grasp.py`

**新增**：
- `interpolate_cartesian_line()` — numpy xyz 插值，姿态锁定  
- `solve_cartesian_ik_chain()` — 链式 IK，30° 跳变防护  
- `execute_cartesian_linear_ik()` — 主入口  
- 初版用 `kinematics_msgs.srv.GetPositionIK`（后改为 `moveit_msgs`）

**同步更新**：`moveit_grasp.md` 终端 8 技术说明（当时写 OMPL+笛卡尔并存）

### 2.3 与 `kuavo_moveit_bridge.py` 冲突分析

- 终端 8 直发 `/kuavo_arm_target_poses` → **勿同时开 bridge**  
- bridge 设计给 RViz / MoveIt 执行器用，与抓取脚本互斥  

---

## 3. 第二阶段：文档重写与操作习惯

### 3.1 按终端 1→8 重写 `moveit_grasp.md`

**用户需求**：在上位机运行终端，能看 `rqt_image_view`。

**改动**：
- 连续编号 1～8（消除旧版跳号）  
- 标明 🤖 下位机 SSH / 🖥️ 上位机本地  
- 新增终端 7 专门 `rqt_image_view`  
- 总览表 + 窗口布局建议  

---

## 4. 第三阶段：五代程序 Bug 审查（过快/不准/障碍）

### 4.1 用户担忧

「手臂运行过快、抓不准、显示有障碍物、无法运行」——前几代程序的通病。

### 4.2 审查发现的 5 个高危缺陷与修复

| # | 问题 | 根因 | 修复（代码常量/函数） |
|---|------|------|----------------------|
| 1 | **手臂过快** | OMPL 输出 80～150 点，`dt≈0.023s`，MPC 猛冲 | `MAX_STREAM_POINTS=35`、`_downsample_joint_chain()`、`MIN_SEGMENT_DT=0.08`、批量下发轨迹 |
| 2 | **抓不准** | IK seed 用延迟的 `/joint_states` 反馈 | `last_commanded_joints_rad` 作 seed；`SETTLE_TIME_SEC=0.6` |
| 3 | **伪障碍** | `clear_octomap` 只调 `/clear_octomap`，MoveIt 实际用 `/move_group/clear_octomap` | `OCTOMAP_CLEAR_SERVICES` 列表依次尝试 |
| 4 | **规划过激** | 默认 velocity scaling=1.0 | `VELOCITY_SCALING=0.35`、`ACCELERATION_SCALING=0.35` |
| 5 | **到位前就下一步** | 无 settle | `wait_joints_settle()` 每步后调用 |

**说明**：审查时不能声称「100% 无 bug」，必须上真机验证。

---

## 5. 第四阶段：网络拓扑与上下位机角色

### 5.1 「怎么在下位机里跑上位机命令」

**解答**：下位机 `ssh leju_kuavo@192.168.26.12`，登录后执行 `kuavo_ros_application` 内命令。  
`rqt_image_view` 需上位机本地显示器，SSH 无 X11 时看不到 GUI。

### 5.2 IP 混淆事件

用户在下位机执行 `arp -a` 看到 `192.168.26.12`，曾误以为下位机是 `.12`。

**最终确认**：
- 下位机 `lab@NUC11TNKi7`：`192.168.26.1/24` on `enx00e04c68345c`  
- 上位机：`192.168.26.12`  

### 5.3 「lab 就是下位机，文档视角是上位机」

**澄清**：`moveit_grasp.md` 默认操作者坐在**上位机**，SSH 进下位机跑控制节点。用户坐在下位机时，命令方向相反（SSH 到 `.12` 开相机）。

---

## 6. 第五阶段：终端 8 启动失败与通信

### 6.1 `kinematics_msgs` ImportError

**现象**：`python3 moveit_auto_grasp.py` 启动即崩溃。  
**解决**：改为 `from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest`。

### 6.2 「之前都可以通信，这次第一次遇到」

**参考**：`auto_grasp_TF2.py` 不用 MoveIt，但同样订阅 `/vla/yolo_target`，证明跨机 ROS 通信正常。  
**MoveIt 版额外依赖**：终端 2 `/joint_states`、终端 3 `/compute_ik`。

### 6.3 启动顺序对齐 auto_grasp_TF2

**问题**：MoveIt 在视觉前加载，长时间阻塞，用户以为卡死。  
**解决**：

```
init 归位 → 视觉 10 帧 → 再加载 MoveGroupCommander
```

与 `auto_grasp_TF2.py` 一致。

### 6.4 「MoveIt 里设置的初始角度被废弃了吗？」

**答**：没有。`execute_dual_arm_init_home()` 使用与 `auto_grasp_TF2` 相同的 `DUAL_ARM_INIT_DEG`（肩 20°、肘 -30°）。  
MoveIt 的 `set_named_target("ready")` 仅用于异常收手路径，**不是**主抓取路径的第一步。

---

## 7. 第六阶段：双臂安全与失败残留姿态

### 7.1 失败后右手前伸、下次抓左手导致重心前倾

**现象**：上次失败右手留在前方；本次抓左手只动左臂，机器人前倾。  
**根因**：早期只对活动臂归位。

**解决演进**：
1. `return_both_arms_to_ready()` — 双臂 OMPL 同步归位  
2. 后改为启动/失败均 `execute_dual_arm_init_home()`  
3. `_freeze_inactive_arm()` — IK/下发时非活动臂锁 init  

### 7.2 「再运行胳膊会回去吗？」

**答**：当前版启动时会 `execute_dual_arm_init_home()`；若上次异常退出，**需重新跑终端 8** 或手动发 init。

---

## 8. 第七阶段：路径几何 — 斜插/先上后前碰瓶

### 8.1 OMPL 预瞄斜插碰倒瓶

**现象**：坐标正确，但路径从下方/侧方斜插，未水平前插。  
**根因**：`smart_execute()` + OMPL 走关节空间最短路，非肩→瓶直线。

**解决**：放弃 OMPL 做预瞄→抓握；改 IK 逆推（先抓握点，再预瞄点）+ `execute_single_pose`。

### 8.2 「先往上升再往前，上升时就碰倒瓶」

**现象**：init 直跳预瞄时，关节空间路径先抬肘。  
**根因**：缺少 auto_grasp 的「曲肘护胸」中间构型。

**解决**：**恢复**流程：

```
曲肘护胸 → 退至预瞄 12cm → 单段水平插入
```

（曾尝试删除曲肘减冗余，实测碰瓶，已撤销。）

### 8.3 12cm 插入方式演进

| 尝试 | 结果 | 当前 |
|------|------|------|
| 手搓 25 点链式 IK | 曾 26 点全无解 / 连杆上翘 | 保留代码备用 |
| `GetCartesianPath` ROS 服务 | 曾 100% 覆盖 | 曾用，后改 |
| **预瞄→抓握单段关节插值** | 与 auto_grasp 一致，不翘连杆 | **当前主路径** |

代码注释：`# 预瞄与抓握共线同高：单段 14 轴插值`

---

## 9. 第八阶段：MoveIt 价值质疑与 IK 全面无解

### 9.1 「MoveIt 只是 IK 求解器，还不如不用？」

**对话结论（经典版）**：**有意为之**。OMPL 插入/收手在 URDF 无桌面碰撞、瓶身在点云里的条件下不可靠；MoveIt 价值在 TRAC-IK + TCP 正确 link。  
**点云版**补充：抬升后 OMPL+OctoMap 收手才有「完整 MoveIt」体验。

### 9.2 链式 IK 26 航点全无解

**多重根因**：
1. RobotState 只传 14 轴 → TRAC-IK 失败  
2. `position` 为 tuple → `TypeError`  
3. 用了 `zarm_r7_link` 而非 `zarm_r7_end_effector`（~17cm Z 误差）  
4. seed 用 init（肘 -30°）而非 official（肘 -90°）  

**修复**：
- `_build_robot_state_seed()` 全身 state + `list(position)`  
- `_ik_group_profile()` → `*_end_effector`  
- `OFFICIAL_IK_SEED_7 = [0,0,0,-π/2,0,0,0]`  
- `/compute_ik` 服务回退链  

---

## 10. 第九阶段：实机微调 — TCP / 收手碰桌 / 非活动臂误动

### 10.1 右爪偏右、抓不准

**原因**：URDF `end_effector` 已含 Y±0.03m，与 `TCP_OFFSET_Y` 叠加需标定。  
**演进**：

| 轮次 | TCP_OFFSET_X | TCP_OFFSET_Y_RIGHT | 备注 |
|------|--------------|-------------------|------|
| vla 原版 | +0.005 | 0.03 | auto_grasp_TF2 |
| 第 1 轮 | +0.005→-0.010 | 0.04→0.052 | 偏前偏右 |
| 第 2 轮 | -0.016 | 0.058 | 仍略偏 |
| 用户改乱 | 回到 0.005/0.04 | 需恢复 | |
| **当前** | **左右分参** | 见 §15 | `tcp_offsets_for_arm()` |

### 10.2 收手碰桌

**根因**：OMPL 收手 + URDF 无桌面 collision。  
**解决**：`execute_vla_style_return()` — 抬升 → 肩膀外摆 75° → 曲肘 → init（来自 `vla_auto_grasp_daemon.py`）。

### 10.3 右手抓、左手跟着动

**解决**：`_freeze_inactive_arm()` + `_build_ik_seed_for_pose()` 非活动臂用 init。

### 10.4 其他参数对齐 vla

- `SAFE_LOCKED_Z`: 0.37 → **0.385**  
- `LIFT_HEIGHT`: **0.22**（22cm）  
- `TCP_OFFSET_Y_LEFT` 历史标定曾至 **0.020**；**当前仓库为 0.004**（见 §15.2）

---

## 11. 第十阶段：OctoMap 可行性讨论与实现

### 11.1 用户问：MoveIt 能否用上位机深度相机 OctoMap 自动避障？

**结论**：原理可行；本仓库最初 **未接通**（`sensors_3d.yaml` 为 `sensors: []`）。

### 11.2 水瓶被识别成障碍物怎么办？

**讨论结论（标准 MoveIt pick）**：
1. 抓取前：YOLO mask 滤除瓶体点云再建图  
2. 或：`CollisionObject` 建模瓶子 + ACM 忽略  
3. 抓取后：`attachObject` 随手动  
4. 本项目的简化方案：抓取前 **`clear_octomap`**，收手时再建图  

### 11.3 用户要求：保留经典版，新建 OctoMap 版

**下位机新增**：

| 文件 | 作用 |
|------|------|
| `moveit_octomap_grasp.py` | 点云版主程序，import `moveit_auto_grasp as mag` |
| `move_group_octomap.launch` | move_group + sensor_manager_octomap |
| `sensor_manager_octomap.launch.xml` | 加载 sensors_3d_octomap.yaml |
| `sensors_3d_octomap.yaml` | 订阅 `/camera/depth/color/points` |
| `launch/depth_to_pointcloud.launch` | 下位机备用点云（一般用上位机内置） |
| `biped_s49_moveit_sensor_manager.launch.xml` | 补 `moveit_sensor_manager` param |

**架构**：
- **阶段 A**：与经典版相同硬编码抓取（防碰瓶）  
- **阶段 B**：抬升后 OctoMap + 桌面盒 + OMPL 收手；失败 → vla 外摆  

**默认 `sensors_3d.yaml` 仍为空** — 必须用 `move_group_octomap.launch`，不能用普通 `move_group.launch` 做点云收手。

---

## 12. 第十一阶段：点云话题存在但无数据

### 12.1 现象

```bash
rostopic list | grep points   # 有 /camera/depth/color/points
rostopic hz .../color/points  # no new messages
rostopic hz .../image_raw     # ~22 Hz OK
rostopic hz /vla/yolo_target  # ~8.7 Hz OK
```

### 12.2 根因

- `/vla/yolo_target` 与点云是**独立管线**  
- Orbbec `gemini_330_series.launch` 默认 `enable_point_cloud:=false`  
- 话题名可能被驱动注册但无人发布  

### 12.3 下位机临时方案

`moveit_octomap_grasp.py` 增加：
- `POINT_CLOUD_CANDIDATES` 多话题探测  
- `resolve_pointcloud_topic()`  
- 无点云时**降级模式**（仅桌面盒 + vla 收手）  
- 明确报错文案区分 YOLO vs 点云  

---

## 13. 第十二阶段：上位机 Cursor 改相机（提示词 + 具体改动）

### 13.1 为什么需要在上位机改？

下位机 Cursor 工作区是 `kuavo-ros-opensource`，**看不到** `kuavo_ros_application`。  
相机 launch、Orbbec 驱动参数在上位机，必须：
- 上位机单独开 Cursor Remote SSH，或  
- 下位机 SSH 改上位机文件  

### 13.2 双 Cursor 工作流

| 窗口 | 连接 | 工作空间 | 改什么 |
|------|------|----------|--------|
| 下位机 Cursor | NUC / lab | `kuavo-ros-opensource` | 抓取脚本、MoveIt launch |
| 上位机 Cursor | `leju_kuavo@192.168.26.12` | `kuavo_ros_application` | 相机 launch、YOLO |

下位机也可 `ssh leju_kuavo@192.168.26.12` 或 `sshfs` 挂载。

### 13.3 发给上位机 Cursor 的完整提示词（原文）

```markdown
## 背景（请先读清楚再改）

我是 Kuavo 人形机器人双臂抓取项目，ROS1 Noetic，双机架构：

- **上位机（本机，当前窗口）**：`leju_kuavo@192.168.26.12`，工作空间 `~/kuavo_ros_application`
  - 终端 5：深度相机 `load_robot_head.launch`
  - 终端 6：YOLO + TF2 视觉，发布 `/vla/yolo_target`
- **下位机**：`lab@192.168.26.1`，工作空间 `~/kuavo-ros-opensource`
  - MoveIt OctoMap 版抓取：`moveit_octomap_grasp.py`
  - OctoMap 订阅点云话题：`/camera/depth/color/points`（见下位机 `sensors_3d_octomap.yaml`）

两台机器共用 `ROS_MASTER_URI=http://kuavo_master:11311`。

## 当前问题（已用 rostopic 验证）

| 话题 | 状态 |
|------|------|
| `/camera/depth/image_raw` | ✅ ~22 Hz，深度正常 |
| `/vla/yolo_target` | ✅ ~8.7 Hz，YOLO 正常 |
| `/camera/depth/color/points` | ❌ 话题存在，但 `rostopic hz` 显示 **no new messages** |

结论：不是下位机 MoveIt 的问题，是**上位机相机 launch 没有真正发布点云**。

## 你的任务（只改上位机本仓库）

请在本工作空间 `kuavo_ros_application` 内完成点云发布修复，要求：

### 1. 查清 launch 链路

- `src/dynamic_biped/launch/load_robot_head.launch`（`use_orbbec:=true`）
- `src/dynamic_biped/launch/orbbec_sensor_robot_enable.launch`
- `src/OrbbecSDK_ROS1/launch/gemini_330_series.launch`
  - 注意：`enable_point_cloud` 默认 `false`
  - 注意：remap `depth/color/points` → `depth_registered/points`

### 2. 修改目标

- 让 `/camera/depth/color/points` 有稳定数据（≥15 Hz）
- 优先在 `orbbec_sensor_robot_enable.launch` 传参，不要大改 SDK 源码
- 若驱动 remap 后发到 `/camera/depth_registered/points`，二选一：
  - A) 改 remap 保持 `/camera/depth/color/points`（推荐）
  - B) `topic_tools/relay` 转发
- 若无彩色点云需求，xyz 即可

### 3. 可选：备用 launch `depth_to_pointcloud.launch`

### 4. 不要动下位机 `kuavo-ros-opensource`、参考脚本、YOLO 逻辑

## 验收

rostopic hz /camera/depth/color/points
rostopic info /camera/depth/color/points
```

### 13.4 上位机 Cursor 的排查结论

1. `load_robot_head.launch` → `orbbec_sensor_robot_enable.launch` → `gemini_330_series.launch`  
2. **`enable_point_cloud` 默认 false** — 根因  
3. 驱动实际发布 **`/camera/depth/points`**（非 color/points）  
4. remap 只影响彩色点云路径；未开 `enable_colored_point_cloud` 时对 MoveIt 无帮助  
5. RealSense 版 `sensor_robot_enable.launch` 有 `enable_pointcloud:=true`，Orbbec 之前漏配  

### 13.5 上位机 Cursor **第一版改动**（relay 方案）

**文件**：`orbbec_sensor_robot_enable.launch`、`orbbec_sensor_only_enable.launch`

```xml
<!-- 主路径 -->
<arg name="enable_point_cloud" value="true" />
<node pkg="topic_tools" type="relay" name="depth_points_to_color_points"
      args="/camera/depth/points /camera/depth/color/points" />
```

**新增**：`dynamic_biped/launch/depth_to_pointcloud.launch`（备用）

**效果**：点云可用，但后续多轮测试出现 **SIGSEGV**（见 §16）。

### 13.6 下位机 agent **第二版稳定改动**（SSH 修订）

**原因**：relay + 驱动内点云 + YOLO + move_group OctoMap 多订阅 → Orbbec 驱动崩溃。

**文件**：`~/kuavo_ros_application/src/dynamic_biped/launch/orbbec_sensor_robot_enable.launch`

| 项 | 改前 | 改后 |
|----|------|------|
| `enable_point_cloud` | `true`（第一版） | **`false`** |
| `color_fps` | 60 | **30** |
| 点云来源 | 驱动 + relay | **`depth_image_proc/point_cloud_xyz` nodelet** |
| 输出话题 | relay | **`/camera/depth/color/points`** |
| node 名 | `depth_points_to_color_points` | **`vla_depth_to_pointcloud`** |

**同步修改**：`orbbec_sensor_only_enable.launch`（同样逻辑）

**launch 链路**（最终）：

```
load_robot_head.launch (use_orbbec:=true)
  └─ orbbec_sensor_robot_enable.launch
       ├─ gemini_330_series.launch (enable_point_cloud:=false, color_fps:=30)
       ├─ nodelet manager vla_pc_manager
       ├─ depth_image_proc/point_cloud_xyz → /camera/depth/color/points
       └─ apriltag_ros continuous_detection
```

---

## 14. 第十三阶段：OctoMap 版实机调试

### 14.1 `moveit_octomap_grasp.py` IndentationError

**现象**：`import moveit_auto_grasp as mag` 失败，line 731 缩进错误。  
**根因**：`smart_execute()` 内 `arm_group.plan()` 多缩进一级。  
**解决**：修正缩进。

### 14.2 「15s 内未收到点云」但 YOLO 正常

**解释**：脚本探测点云；YOLO 与点云无关。后改为多话题 `POINT_CLOUD_CANDIDATES` + 降级模式。

### 14.3 Ready to take commands 后卡 ~20s

**根因**：`wait_for_octomap(20)` 空等 `/octomap_full`。  
**解决**：`probe_octomap_ready(2.5s)`；MoveIt 常在内部建图不发布 `/octomap_full`，收手 OMPL 成功即说明 OctoMap 有效。

### 14.4 OctoMap 首帧超时 WARN（误报）

**现象**：WARN 但 `[OctoMap外移] OMPL 成功`。  
**结论**：误报，已改 INFO。

### 14.5 OctoMap 版实机成功日志特征

```
✅ 点云已锁定: /camera/depth/color/points
🚀 阶段A [硬编码抓取] ...
🗺️ 阶段B：OctoMap + 桌面盒 碰撞感知收手...
✅ [OctoMap外移(退12cm)] OMPL 成功
```

---

## 15. 第十四阶段：TCP 多轮标定与 IK WARN

### 15.1 多轮微调记录

| 用户反馈 | 调整 |
|----------|------|
| 偏右 + 偏前 | X: -0.010, Y_RIGHT: 0.052 |
| 仍略偏 | X: -0.016, Y_RIGHT: 0.058 |
| 用户改乱文件 | 恢复并改为左右分参 |
| 右手略偏前、左手略偏左 | X_RIGHT: -0.024, Y_LEFT: 0.020, Y_RIGHT: 0.065 |

### 15.2 当前 TCP（`tcp_offsets_for_arm()`）

```python
TCP_OFFSET_X_LEFT  = -0.013
TCP_OFFSET_X_RIGHT = -0.022
TCP_OFFSET_Y_LEFT  = 0.004   # 偏左 → 减小
TCP_OFFSET_Y_RIGHT = 0.065   # 偏右 → 增大
SAFE_LOCKED_Z      = 0.385
LIFT_HEIGHT        = 0.22
LIFT_HEIGHT_FALLBACKS_M = (0.22, 0.18, 0.14, 0.10)
PRE_GRASP_DIST     = 0.12
```

**调参规则**：
- 偏前 → X 更负  
- 右手偏右 → Y_RIGHT 增大  
- 左手偏左 → Y_LEFT 减小  

---

## 16. 第十五阶段：Orbbec 相机 SIGSEGV 卡死

### 16.1 现象

跑 1～2 轮抓取后，上位机终端 5 相机挂：
```
Received signal: 11
usbdevfs reset ...
boost: mutex lock failed in pthread_mutex_lock: Invalid argument
Received signal: 6
[camera/camera-1] process has died [pid ..., exit code 11]
```

右下脚 `rostopic hz` 全部 `no new messages`。

### 16.2 根因

- 非抓取脚本直接杀相机，而是 **Orbbec 驱动进程崩溃**  
- 崩溃前日志：`point cloud subscribed`、`Image stream color subscribed`  
- 第一版 `enable_point_cloud:=true` + relay + move_group OctoMap 订阅 + YOLO + apriltag → 多订阅压力 + 驱动 bug  
- USB 复位与崩溃线程竞态 → mutex 二次 abort  

### 16.3 解决

见 [§13.6](#136-下位机-agent-第二版稳定改动ssh-修订) depth_image_proc 方案。  
**操作**：Ctrl+C 重启终端 5；经典版比 OctoMap 版更省相机负载。

### 16.4 上位机曾出现 Decode frame failed / MJPG conversion failed

早期日志提示深度流可能部分损坏；与 SIGSEGV 可独立出现。深度 `image_raw` 正常时，depth_image_proc 仍可生成点云。

---

## 17. 第十六阶段：「MoveIt 真在用吗？」与启动 WARN

### 17.1 用户疑问

跑 `moveit_auto_grasp.py` 轨迹与 vla/TF2 差不多，是否没用 MoveIt？

### 17.2 答案

**经典版 MoveIt 只做**：
1. 加载模型  
2. `/compute_ik` 算关节角  
3. 异常时 OMPL（主路径不用）  

**不做**：OMPL 抓取、点云收手。  
日志：`肩膀外摆避障` 结尾 = 无点云。

### 17.3 启动 WARN 详解

| WARN | 含义 |
|------|------|
| `Link xxx has visual geometry but no collision geometry` | URDF 无 collision mesh；腿/臂/头/相机/radar 均如此 |
| `IK plugin relies on deprecated API` | TRAC-IK 插件接口旧 |
| `No root/virtual joint in SRDF` | 正常默认 |

**影响**：经典版可忽略；OctoMap 版 OMPL 质量受 URDF 无碰撞体限制，主要靠点云+桌面盒。

---

## 18. 第十七阶段：「记得有点云收手」记忆混淆

### 18.1 用户记忆

「之前手臂回来用点云避障，现在全程没用点云。」

### 18.2 解释

| 运行脚本 | 收手方式 | 点云 |
|----------|----------|------|
| `moveit_auto_grasp.py` | vla 肩膀外摆 | ❌ 全程 |
| `moveit_octomap_grasp.py` | 阶段 B OMPL+OctoMap | ✅ 仅抬升后 |

用户最近日志结尾为 `肩膀外摆避障` → 跑的是**经典版**。  
点云版需终端 3 `move_group_octomap.launch` + 终端 8 `moveit_octomap_grasp.py`。

---

## 附录 A：下位机全部文件变更清单

### A.1 主程序

| 文件 | 变更摘要 |
|------|----------|
| `moveit_auto_grasp.py` | 从零构建；IK 引擎；双臂安全；TCP 分参；vla 收手；曲肘护胸；单段插入；流式安全参数 |
| `moveit_octomap_grasp.py` | **新建**；阶段 A/B；点云探测；桌面盒；OctoMap 收手；import mag |

### A.2 MoveIt 配置

| 文件 | 变更 |
|------|------|
| `kuavo_arm_moveit_config/launch/move_group_octomap.launch` | **新建** |
| `kuavo_arm_moveit_config/launch/sensor_manager_octomap.launch.xml` | **新建** |
| `kuavo_arm_moveit_config/config/sensors_3d_octomap.yaml` | **新建**，topic=`/camera/depth/color/points` |
| `kuavo_arm_moveit_config/launch/biped_s49_moveit_sensor_manager.launch.xml` | 从空文件补 `moveit_sensor_manager` param |
| `kuavo_arm_moveit_config/config/sensors_3d.yaml` | **保持** `sensors: []`（经典 move_group 不订阅点云） |

### A.3 文档与 launch

| 文件 | 变更 |
|------|------|
| `moveit_grasp.md` | 8 终端手册；双模式；快速复制；MoveIt 实际用途 |
| `question.md` | 本文档 |
| `launch/depth_to_pointcloud.launch` | **新建**（下位机备用） |

### A.4 未改动的参考文件

- `auto_grasp_TF2.py`  
- `vla_auto_grasp_daemon.py`  
- `kuavo_moveit_bridge.py`（仍禁止与终端 8 同开）  

---

## 附录 B：上位机 launch 改前 / 改后

### B.1 改前（Orbbec 默认 + 无点云）

```xml
<include file="$(find orbbec_camera)/launch/gemini_330_series.launch">
    <arg name="color_fps" value="60" />
    <!-- enable_point_cloud 默认 false -->
</include>
```

### B.2 第一版（上位机 Cursor · relay · 会 SIGSEGV）

```xml
<arg name="enable_point_cloud" value="true" />
<node pkg="topic_tools" type="relay" name="depth_points_to_color_points"
      args="/camera/depth/points /camera/depth/color/points" />
```

### B.3 最终稳定版（depth_image_proc）

```xml
<include file="$(find orbbec_camera)/launch/gemini_330_series.launch">
    <arg name="color_fps" value="30" />
    <arg name="enable_point_cloud" value="false" />
</include>
<node pkg="nodelet" type="nodelet" name="vla_pc_manager" args="manager" />
<node pkg="nodelet" type="nodelet" name="vla_depth_to_pointcloud"
      args="load depth_image_proc/point_cloud_xyz vla_pc_manager --no-bond">
    <remap from="image_rect" to="/camera/depth/image_raw" />
    <remap from="camera_info" to="/camera/depth/camera_info" />
    <remap from="points" to="/camera/depth/color/points" />
</node>
```

**路径**：`~/kuavo_ros_application/src/dynamic_biped/launch/orbbec_sensor_robot_enable.launch`

---

## 附录 C：发给上位机 Cursor 的完整提示词

见 [§13.3](#133-发给上位机-cursor-的完整提示词原文)。

---

## 附录 D：标准 MoveIt Pick 与水瓶伪障碍

### D.1 工业界标准流程

1. 感知 → 点云/深度  
2. **分割/滤除**目标物体点云（YOLO mask）  
3. OctoMap 只含环境障碍  
4. 规划 approach → grasp  
5. `attachObject` 后物体随手动  
6. 规划 retreat  

### D.2 本项目简化做法

| 阶段 | 做法 |
|------|------|
| 抓取 | 硬编码 + 插入前 `clear_octomap`（防瓶身挡 12cm） |
| 收手 OctoMap 版 | 抬升后建图 + 桌面 `CollisionObject` + OMPL |
| 收手经典版 | vla 肩膀外摆（不依赖点云） |

### D.3 未实现但讨论过的方向

- YOLO mask → 过滤点云再 OctoMap  
- 水瓶作 `CollisionObject` + AllowedCollisionMatrix  
- URDF 补手臂 collision mesh  

---

## 附录 E：问题—原因—解决 速查表（全）

| # | 问题 | 根因 | 解决 | 状态 |
|---|------|------|------|------|
| 1 | kinematics_msgs ImportError | 包/API 错误 | moveit_msgs | ✅ |
| 2 | 链式 IK 全无解 | 14 轴 state | 全身 RobotState | ✅ |
| 3 | tuple 赋值崩溃 | position 为 tuple | list() | ✅ |
| 4 | 抓握 IK 无解 | wrist 非 TCP ~17cm | *_end_effector | ✅ |
| 5 | 单臂不归位重心前倾 | 只收活动臂 | 双臂 init | ✅ |
| 6 | 视觉有臂不动 | MoveIt 阻塞顺序 | 先视觉后 MoveIt | ✅ |
| 7 | OMPL 斜插碰瓶 | 关节最短路 | IK+单段插入 | ✅ |
| 8 | Python cartesian 闪退 | Boost bug | 禁 Python 绑定 | ✅ |
| 9 | 手臂过快 | OMPL 上百点 dt 极小 | 降采样+MIN_SEGMENT_DT | ✅ |
| 10 | 抓不准(延迟) | seed 用反馈非指令 | last_commanded_joints | ✅ |
| 11 | 伪障碍清不掉 | 错 service 名 | OCTOMAP_CLEAR_SERVICES | ✅ |
| 12 | 收手碰桌 | 无桌面 collision | vla 外摆 | ✅ |
| 13 | 左手误动 | IK 改非活动臂 | _freeze_inactive_arm | ✅ |
| 14 | init→预瞄扫瓶 | 缺曲肘 | 恢复曲肘护胸 | ✅ |
| 15 | 删曲肘冗余 | 实测碰瓶 | 恢复曲肘 | ✅ |
| 16 | 右爪偏右 | TCP 标定 | Y_RIGHT 增大 | ✅ 可微调 |
| 17 | 偏前 | TCP X | X 更负 | ✅ 可微调 |
| 18 | compute_ik WARN | 服务名顺序 | 优先 /compute_ik | ✅ |
| 19 | MoveIt 只是 IK? | 有意设计 | 见 §17 | ✅ 设计 |
| 20 | OctoMap 未接通 | sensors 空 | octomap launch+yaml | ✅ |
| 21 | 点云话题无数据 | enable_point_cloud false | depth_image_proc | ✅ |
| 22 | 15s 未收到点云 | 探测逻辑 | 多话题+降级 | ✅ |
| 23 | OctoMap 首帧超时 20s | 误等 /octomap_full | probe 2.5s | ✅ |
| 24 | IndentationError | 缩进 | 修 smart_execute | ✅ |
| 25 | 相机 SIGSEGV | 驱动点多云+多订阅 | 关驱动点云 | ✅ |
| 26 | mutex lock failed | USB 复位竞态 | 重启终端 5 | ✅ |
| 27 | 以为全程点云 | 跑错脚本 | OctoMap 3+8 | 📋 操作 |
| 28 | 上下位机 IP 混淆 | .12 vs .1 | 见 §5.2 | ✅ |
| 29 | 下位机看不到上位机代码 | 双工作空间 | Remote SSH | ✅ |
| 30 | bridge 冲突 | 同话题 | 勿同开 | 📋 操作 |
| 31 | URDF 无 collision WARN | 未建模 | 可忽略/待补 | 📋 |
| 32 | 水瓶伪障碍 | 点在 OctoMap | clear_octomap | ✅ |
| 33 | 用户改乱 TCP | 手动改文件 | 恢复分参 | ✅ |
| 34 | nodelet_manager update rate | 负载 | 降 fps | ✅ |
| 35 | camera 标定 yaml 缺失 | 无 camera_info 文件 | 驱动用默认内参 | ⚠️ |
| 36 | apriltag 同 launch | 额外 CPU | 与抓取无直接冲突 | ℹ️ |
| 37 | 双 publisher 点云 | relay+depth_to_pointcloud 同开 | 只开一种 | 📋 |
| 38 | 经典版不订阅点云 | move_group.launch | 设计如此 | ✅ |
| 39 | GetCartesianPath 备用 | 主路径改单段 | 代码仍保留 | ℹ️ |
| 40 | smart_execute 未调主路径 | 有意 | 仅异常/遗留 | ✅ |

---

## 附录 F：版本演进与当前稳定流程

### F.1 版本表

| 版本 | 插入 | 收手 | 点云 | IK |
|------|------|------|------|-----|
| v0 | OMPL 全程 | OMPL | ❌ | kinematics_msgs❌ |
| v1 | 手搓 25 点 IK | OMPL | ❌ | 14轴 state❌ |
| v2 | +end_effector | 双臂 OMPL | ❌ | moveit_msgs ✅ |
| v3 | GetCartesianPath 服务 | vla 外摆 | ❌ | ✅ |
| v4 | 单段关节插值 | vla 外摆 | ❌ | ✅ |
| v5 | +曲肘护胸恢复 | vla 外摆 | ❌ | ✅ |
| **v6 经典** | 同 v5 | vla | ❌ | 左右 TCP 分参 |
| **v7 OctoMap** | 同 v6 阶段 A | OMPL+OctoMap | ✅ 阶段 B | 同 v6 |

### F.2 经典版完整流程

```
wait /joint_states
→ arm_traj_change_mode(2)
→ 夹爪张开
→ execute_dual_arm_init_home()          # DUAL_ARM_INIT_DEG
→ _collect_vision_targets_tf2_style()   # 10 帧 /vla/yolo_target
→ MoveGroupCommander left/right_arm
→ tcp_offsets_for_arm + /compute_ik     # 抓握、预瞄、抬升
→ 曲肘护胸 execute_single_pose
→ 退预瞄 execute_single_pose
→ 水平插入 execute_single_pose          # 单段，非 OMPL
→ 夹爪闭合
→ 抬升 execute_single_pose
→ execute_vla_style_return()            # 肩膀 75° 外摆
→ 松爪 control_mode(0)
```

### F.3 OctoMap 版增量

```
… 同上至抬升 …
→ octomap_retract_after_lift()
    → wait 点云刷新
    → PlanningScene 桌面盒 DESK_BOX_ID
    → OMPL 外移+后退 (avoid_collisions=True)
    → OMPL ready
    → 失败则 execute_vla_style_return()
```

---

## 附录 G：关键函数与参数索引

### G.1 `moveit_auto_grasp.py`

| 函数/常量 | 作用 |
|-----------|------|
| `tcp_offsets_for_arm()` | 左右手 TCP 分参 |
| `_collect_vision_targets_tf2_style()` | 10 帧 TF2 视觉 |
| `_solve_pose_ik()` | MoveIt IK 单点 |
| `execute_single_pose()` | 硬编码单段下发 |
| `execute_dual_arm_init_home()` | init 归位 |
| `execute_vla_style_return()` | vla 收手 |
| `_freeze_inactive_arm()` | 锁非活动臂 |
| `clear_octomap_cache()` | 清瓶身伪障碍 |
| `smart_execute()` | OMPL 规划（**主路径不调**） |
| `execute_cartesian_linear_ik()` | 链式 IK（**备用**） |
| `execute_cartesian_path_service()` | 笛卡尔服务（**备用**） |

### G.2 `moveit_octomap_grasp.py`

| 函数 | 作用 |
|------|------|
| `resolve_pointcloud_topic()` | 自动探测点云话题 |
| `probe_octomap_ready()` | 非阻塞 OctoMap 探测 |
| `setup_desk_collision_box()` | 桌面 CollisionObject |
| `octomap_retract_after_lift()` | 阶段 B 收手 |
| `_run_octomap_grasp_sequence()` | 主流程 |

### G.3 OctoMap 参数

```yaml
# sensors_3d_octomap.yaml
point_cloud_topic: /camera/depth/color/points
max_range: 2.5
max_update_rate: 2.0
```

```python
# moveit_octomap_grasp.py
DESK_BOX_CENTER = (0.42, 0.0, 0.33)
DESK_BOX_SIZE = (1.20, 0.90, 0.06)
AVOID_OUTWARD_Y = 0.10
OCTOMAP_SETTLE_AFTER_LIFT = 2.0
```

---

*本文档覆盖 MoveIt 抓取开发全周期对话内容。遇新问题请追加章节并更新速查表。*
