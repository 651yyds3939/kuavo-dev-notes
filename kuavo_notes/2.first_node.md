# 📝 Kuavo 机器人二次开发笔记：第一课（头部单点控制开始）

**目标**：编写 Python 脚本，通过发号施令，让 Gazebo 仿真环境中的机器人头部完成“向左转 -> 向右转 -> 回正”的动作闭环。

## 一、 动手前必做的准备：“查阅字典”
[cite_start]在写任何控制代码前，第一步永远是去官方的《接口使用文档》中寻找对应的“填表说明” [cite: 66]。

针对头部控制，我们需要明确以下三个核心要素：
1. [cite_start]**往哪里发（话题名称）**：`/robot_head_motion_data` [cite: 66]
2. [cite_start]**用什么格式发（消息类型）**：`kuavo_msgs/robotHeadMotionData` [cite: 66]
3. **安全边界是什么（参数范围）**：
   * [cite_start]偏航角 (Yaw/左右)：`[-30, 30]` 度 [cite: 66]
   * [cite_start]俯仰角 (Pitch/上下)：`[-25, 25]` 度 [cite: 66]

## 二、 代码编写的“四步走”逻辑

在 ROS 1 (`rospy`) 中编写一个发布者节点，其标准逻辑永远可以拆解为以下四个步骤：

### 第 1 步：导入官方模板与初始化节点
* **动作**：引入刚才查到的 `robotHeadMotionData` 消息包。
* **动作**：使用 `rospy.init_node()` 给当前的脚本上个户口，告诉系统“我来了，我叫 `head_controller`”。

### 第 2 步：创建发布者（架设天线）
* **动作**：声明一个发布者对象。必须精确填入我们在第一步查到的**话题名称**和**消息类型**。
* **避坑关键**：创建完毕后，**必须加上 `rospy.sleep(1.0)`**。网络拓扑的建立需要时间，如果不延时直接发数据，底层的电机节点大概率还未连上，会导致第一条指令丢失。

### 第 3 步：构造包裹与填入数据
* **动作**：实例化一个消息空壳：`msg = robotHeadMotionData()`。
* **动作**：在其唯一的属性栏 `joint_data` 中填入具体的角度数组 `[Yaw, Pitch]`。

### 第 4 步：发射与动作延时
* **动作**：调用 `publish(msg)` 将数据打入网络。
* **避坑关键**：发布指令瞬间就完成了，但**机器人的物理电机转动需要时间**。因此每次发完指令后，必须加上 `rospy.sleep(2.0)`，强制脚本暂停执行，留出充足的物理运动时间，然后再发下一个指令。

## 三、 完整代码模板

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
# 1. 导入官方定义好的头部控制消息类型
from kuavo_msgs.msg import robotHeadMotionData

def move_head_logic():
    # 步骤 1：初始化节点 (上户口)
    rospy.init_node('my_head_controller', anonymous=True)
    
    # 步骤 2：创建发布者 (架天线)并等待连接
    pub = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
    rospy.sleep(1.0) # 【关键】等待底层订阅者连上频道
    
    # 步骤 3：构造消息对象 (拿空纸箱)
    head_msg = robotHeadMotionData()
    
    # 步骤 4：填入数据并发布执行 (动作序列)
    
    # 动作 A：向左看
    head_msg.joint_data = [30.0, 0.0]  # [Yaw, Pitch]
    pub.publish(head_msg)
    rospy.sleep(2.0) # 【关键】给物理电机留出转动时间
    
    # 动作 B：向右看
    head_msg.joint_data = [-30.0, 0.0]
    pub.publish(head_msg)
    rospy.sleep(2.0)
    
    # 动作 C：回正
    head_msg.joint_data = [0.0, 0.0]
    pub.publish(head_msg)
    
if __name__ == '__main__':
    try:
        move_head_logic()
    except rospy.ROSInterruptException:
        pass
```

## 四、 部署与运行规范
在终端中执行以下操作，确保代码能够被系统正确识别并运行：
1. **进入工作空间并刷新环境**：
   ```bash
   docker exec -it -e DISPLAY=$DISPLAY kuavo_container /bin/zsh 
   cd ~/kuavo_ws
   source devel/setup.zsh
   mkdir -p ~/kuavo_ws/src/my_first_package
    cd ~/kuavo_ws/src/my_first_package
   ```
2. **编写脚本并赋予脚本最高级别的执行权限**：
   ```bash
   nano head_control.py 
   chmod +x my_head_control.py
   ```
3. **运行脚本**：
   ```bash
   python3 my_head_control.py
   ```


## 五、 深层原理：通讯“四大金刚”的关系



1.  **节点 (Node)**：**干活的实体**。你的 Python 脚本是寄件人节点，底层的 C++ 控制器是收件人节点。
2.  **话题 (Topic)**：**快递通道**。它是虚拟的滑梯，连接着发布者和订阅者。
3.  **消息 (Message)**：**官方机密表格**。例如 `.msg` 文件。它本身没有逻辑，只是一个装载数据的“死”容器。
4.  **大总管 (Master/roscore)**：**婚姻介绍所**。它只负责牵线，数据传输通过节点间点对点完成，不经过大总管中转。

**核心结论**：你修改 `joint_data` 的数值，本质上是在**填表**。底层节点收到表后，触发内部的**回调函数 (Callback)** 才会去真正驱动电机。



## 六、 工程避坑指南：Docker 权限与环境规范

### 1. 文件夹“带小锁头”怎么办？
**现象**：在宿主机修改 Docker 内创建的文件夹时提示“权限不足”或看到锁头图标。
**原因**：Docker 内默认以 `root` 权限创建文件，宿主机普通用户无法直接写入。
**一键解锁命令**：
在 Docker 终端内执行：
```bash
sudo chmod -R 777 ~/kuavo_ws/src/my_first_package
```

### 2. 运行规范
* **权限赋予**：每次新建脚本后，需执行 `chmod +x 脚本名.py` 才能运行。
* **环境刷新**：打开新终端后，务必执行 `source devel/setup.zsh` 否则找不到自定义消息包。
* [cite_start]**降温大法**：仿真时加上 `gui:=false` 可显著降低 CPU 负载 [cite: 70]。


