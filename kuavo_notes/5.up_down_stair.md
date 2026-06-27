
# 🤖 Kuavo 机器人上下楼梯项目技术笔记 4月27日

## 1. 项目背景与环境
本实验基于 **ROS Humble/Noetic** 和 **Gazebo** 仿真环境。目标是让 Kuavo 人形机器人在没有视觉反馈（开环控制）的情况下，通过精确的步态规划完成连续阶梯的攀爬与下降。

### 核心环境启动指令
在进行任何调试前，必须确保仿真环境干净：

```bash
# 1. 彻底清理残留进程（非常重要，防止物理引擎卡顿）
killall -9 gzserver gzclient

# 2. 启动带楼梯参数的 Gazebo 仿真
# 注意：stairs_x 决定了楼梯位置，默认建议 0.15 左右
roslaunch humanoid_controllers load_kuavo_gazebo_sim.launch load_stairs:=true stairs_x:=0.15
```

---

## 2. 关键动力学参数 (Config)
在 `StairClimbingPlanner` 类的 `__init__` 中，以下参数与物理引擎的刚体摩擦和重力高度对齐：

| 参数 | 设定值 | 说明 |
| :--- | :--- | :--- |
| `dt` | `0.6` | 步态周期，决定了迈步的快慢 |
| `step_height` | `0.134` | 物理台阶 0.13m + 4mm 的“虚拟气垫”防止砸地 |
| `step_length` | `0.28` | 台阶深度，决定水平位移 |
| `foot_w` | `0.12` | 机器人半脚长，用于计算重心支撑 |

---

## 3. 核心逻辑与代码位置

### 3.1 动量初始化：前进 5cm
**现象：** 直接起步上楼，机器人常因重心滞后向后翻倒。
**对策：** 在执行上楼前，先让机器人微动。
```python
# plan_move_to(dx=0.05)
# 这 5cm 给了机器人一个向前的初速度和惯性（Momentum），让重心提前压在脚尖。
```

### 3.2 转身逻辑：平台上移
**现象：** 登顶后原地转身，脚后跟容易悬空。
**对策：** 转身前先深入平台。
```python
# 在 __main__ 下楼分支中：
planner.plan_move_to(dx=0.15, dy=0.0, dyaw=0.0) # 先往前走，确保脚完全在平台上
planner.plan_move_to(dx=0.0, dy=0.0, dyaw=180.0) # 然后原地转身
```

### 3.3 下楼步幅微调 (The "-0.01" Magic)
**现象：** 下楼梯时若步子迈太大，腿部伸直进入“奇异点”，容易摔倒；若太小，踩在坎上会穿模。
**特殊代码位置：** `plan_down_stairs` 循环内。
```python
# 经过实测的最优偏移量
first_step_offset = 0.0    # 第一步保持标准
normal_step_offset = -0.01  # 后续每步稍微“收”着点，维持重心在脚心
```

---

## 4. 腾空相轨迹优化 (Swing Phase)
这是防止机器人“陷进楼梯”或“砸地”的核心。

### 智能分流逻辑
在 `plan_swing_phase` 函数中，通过判断 Z 轴位移决定轨迹：

```python
is_downward = next_foot_pose[2] < prev_foot_pose[2] - 0.05

if is_downward:
    # 【下楼软着陆轨迹】
    # 拒绝高抬腿砸地，采用平滑切入
    z_traj = [
        prev_foot_pose[2],
        prev_foot_pose[2] + 0.04,  # 稍微抬起避开棱角
        next_foot_pose[2] + 0.05,  # 提前降落缓冲
        next_foot_pose[2]          # 触地
    ]
else:
    # 【上楼高抬腿轨迹】
    z_traj = [prev_foot_pose[2], ..., base_height + swing_height, next_foot_pose[2]]
```

---

## 5. 常见错误诊断 (Troubleshooting)

### 1. `wbc_planned_joint_acc is out of range`
* **含义：** 全身控制器计算出的关节加速度爆表。
* **原因：** 通常是因为**穿模（Penetration）**。当机器人的脚陷进台阶内部，物理引擎产生巨大的排斥力，导致数学求解器无法收敛。
* **解决：** 检查 `step_height` 是否过低，或重心是否偏移过大。

### 2. 机器人“踉跄”但没摔倒
* **分析：** 这说明你的重心（CoM）控制得很好，虽然物理碰撞上有小瑕疵，但 WBC 的鲁棒性保住了平衡。
* **结论：** 在盲走模式下，这是可以接受的最优状态。

---

## 6. 项目总结与心得
1.  **控制变量法：** 每次只动一个参数（如步长增加 1cm），观察现象，记录结果，不行立刻回滚。
2.  **尊重物理常识：** 机器人不是线条，是有体积的。脚尖踢到台阶立面、脚后跟挂在边缘，都是穿模的元凶。
3.  **动量是王道：** 很多时候摔倒不是高度不对，而是由于没有初始速度导致的“启动冲击”。

---

## 7. 最终运行工作流
```bash
# 执行上楼（默认）
rosrun humanoid_controllers stairClimbPlanner.py

# 执行转身+下楼（带参数）
rosrun humanoid_controllers stairClimbPlanner.py --down_stairs
```

---

