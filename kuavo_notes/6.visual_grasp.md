# Kuavo 人形机器人：AprilTag 视觉抓取全流程指令集 (SOP)

4月26日凌晨：本笔记记录了在 Docker 容器及 Terminator 分屏环境下，运行 Kuavo V42 视觉抓取项目的完整指令流水线。

---

## 1. 环境准备与编译阶段

### 1.1 进入 Docker 环境
在宿主机终端执行：
```bash
# 启动并进入容器 (zsh 环境)
docker start kuavo_container
docker exec -it kuavo_container zsh
```

### 1.2 源码编译 (工作空间根目录)
执行以下命令确保所有底层包已编译：
```bash
# 进入工作空间
cd ~/kuavo_ws

# 编译核心控制器、SDK 与 IK 模块
catkin build humanoid_controllers
catkin build kuavo_sdk
catkin build motion_capture_ik

# 刷新环境变量 (每个新开终端都必须执行此句)
source devel/setup.zsh
```

---

## 2. 核心执行阶段 (Terminator 4 路分屏)

请按照 **1 -> 2 -> 3 -> 4** 的严格顺序启动以下终端：

### 🖥️ 终端 1：底层物理世界 (左上)
拉起物理仿真引擎，启动 WBC 全身控制大脑。
```bash
source devel/setup.zsh
roslaunch humanoid_controllers load_kuavo_gazebo_sim.launch record_bag:=false
```
*💡 注：如果清理过缓存，请在此终端看到 `MPC node is ready` 且 Gazebo 里的桌子/水瓶出现后再操作下一步。*

### 🖥️ 终端 2：运动学计算层 (右上)
启动逆运动学 (IK) 服务节点，等待指令进行 SNOPT 优化求解。
```bash
source devel/setup.zsh
roslaunch motion_capture_ik ik_node.launch
```

### 🖥️ 终端 3：感知注入层 (左下)
运行视觉 Mock 脚本，向系统不断广播目标的 XYZ 坐标。
```bash
source devel/setup.zsh
python3 src/demo/arm_capture_apriltag/mock_tag_publisher.py
```

### 🖥️ 终端 4：业务逻辑控制 (右下)
执行抓取脚本，进行剥夺控制权、IK 求解与动作下发。
```bash
source devel/setup.zsh
# --offset_start False 用于仿真模式；--cost_weight 1.0 用于优化收敛精度
python3 src/demo/arm_capture_apriltag/arm_capture_apriltag.py --offset_start False --cost_weight 1.0
```

---

## 3. 运维与环境清理指令

### 3.1 强制清理“僵尸”进程
当 Gazebo 出现桌子重叠、方块乱飞或节点无法启动时，在任意终端执行：
```bash
killall -9 gzserver gzclient
```

### 3.2 深度清理底层大脑缓存
当修改了 `kuavo.json` 或 URDF 物理参数后，必须执行：
```bash
rm -rf /var/ocs2/kuavo_v42
```

### 3.3 修改配置文件 (VS Code 路径)
直接在宿主机 VS Code 中修改以下文件：
- **物理配置**：`src/kuavo_assets/config/kuavo_v42/kuavo.json`
- **抓取位置**：`src/demo/arm_capture_apriltag/mock_tag_publisher.py`
- **运动姿态**：`src/demo/arm_capture_apriltag/arm_capture_apriltag.py`

---

## 4. 关键调参经验总结 (备忘录)

| 调试目标 | 修正位置 | 建议操作 |
| :--- | :--- | :--- |
| **解决解不收敛 (误差大)** | `arm_capture_apriltag.py` | 增加 `--cost_weight 1.0` |
| **解决臂展极限死锁** | `arm_capture_apriltag.py` | 将 Pitch 从 `-1.57` 改为 `0.0` (平推抓取) |
| **解决目标偏远问题** | `mock_tag_publisher.py` | 将 `X` 坐标缩减至 `0.28` 左右 |
| **解决手指丢包问题** | `arm_capture_apriltag.py` | 在声明 Publisher 后添加 `rospy.sleep(0.5)` |

---

