# kuavo-robot-deploy — RL 部署工作空间

> **本目录说明**：本目录为 [`kuavo-dev-notes`](../../../../README.md) / [`kuavo-rl-opensource`](../../readme.md) 下的部署仓。S49 舞蹈见 [`kuavo_notes/23.*`](../../../../kuavo_notes/23.5.RL_dance_deploy_hybrid.md)。Docker 内路径常为 `/root/kuavo_ws`。

---

# 如何使用
## 确认机器人版本和总质量
#### 机器人版本
- 机器人版本通过环境变量`$ROBOT_VERSION`设置，版本号涉及不同机器人模型、硬件设置等, 需要和自己的机器人匹配。
- 在终端执行`echo $ROBOT_VERSION`查看当前设置的版本号，如果没有设置，通过以下设置版本号(其中的40代表4.0版本，根据实际情况修改)：

   1. 在当前终端执行(临时设置): 

        `export ROBOT_VERSION=40`

   2. 将其添加到你的 `~/.bashrc` 或者 `~/.zshrc` 终端配置文件中:
    如执行: 

        `echo 'export ROBOT_VERSION=40' >> ~/.bashrc`

    添加到 `~/.bashrc` 文件(bash终端)末尾，重启终端后生效

#### 机器人质量
- 由于每台机器人的选配不同，质量也不同，需要确认机器人的总质量，确保模型准确。(出厂时的质量会修改正确一次)
- 机器人总质量存储于`~/.config/lejuconfig/TotalMassV${ROBOT_VERSION}`文件中(${ROBOT_VERSION}为上述设置的版本号)，编译时会自动读取该文件，校准仓库中的模型质量。
- 机器人称重之后，将总质量写入`~/.config/lejuconfig/TotalMassV${ROBOT_VERSION}`文件中即可。
- ocs2中使用了cppad自动微分库，cppad的缓存与模型相关
  - 因此每次修改总质量文件时，会`自动`删除缓存目录`/var/ocs2/biped_v${ROBOT_VERSION}`, 下一次运行时会自动重新编译cppad模型(大概4分钟)
  - 如果手动修改了仓库中的模型，同样需要删除缓存目录，重新编译cppad模型

## 编译
```bash
若是开源仓库：
source installed/setup.bash
catkin build humanoid_controllers
若不是开源仓库，则：
catkin build humanoid_controllers

```
## 参数修改

修改 `../kuavo-ros-opensource/src/kuavo_assets/config/kuavo_v42/kuavo.json`中的第40行（此处修改机器人对应版本的json文件）：
```
"use_anthropomorphic_gait":false,
改为：
"use_anthropomorphic_gait":true,
```

## 运行
```bash
source devel/setup.zsh  #或者source devel/setup.bash
```仿真
roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch # 启动rl控制器、wbc、仿真器
```实物
roslaunch humanoid_controllers load_kuavo_real.launch cali:=true cali_arm:=true # 可以选择cali:=true 进行标定，cali_arm:=true 进行机械臂标定。
```
可以通过修改launch文件wbc_frequency、sensor_frequency参数来调整控制器、传感器的频率，同时在launch文件中指定了rl_param文件和onnx网络模型文件的位置。
```xml
    <arg name="wbc_frequency"       default="1000"/>
    <arg name="sensor_frequency"    default="1000"/>
    <arg name="rl_param"  default="$(find humanoid_controllers)/config/kuavo_v$(arg robot_version)/rl/rl_param.info"/>
    <arg name="network_model_file" default="$(find humanoid_controllers)/model/networks"/>

```
## 网络模型文件
- 网络模型文件存储于`$(find humanoid_controllers)/model/networks`目录下，文件名称为`${机器人版本}_model_${模型特征}_${日期}.onnx`，可以根据实际情况进行修改。

## 参数文件
- 参数文件储存于：`$(find humanoid_controllers)/config/kuavo_v${机器人版本}/rl/<参数文件名称>.info`，可以修改rl_param.info文件中的参数，包括神经网络的onnx文件路径、输入输出维度等，具体可以参考该文件中的注释。
## 特别说明
- rl_param.info文件中可以定义singleInputData：
```
singleInputData
{
    commandPhase
    {
        startIdx    0
        numIdx      2
        obsScales   1.0
    }
    command
    {
        startIdx    0
        numIdx      4
        obsScales   1.0
    }
    jointPos
    {
        startIdx    0
        numIdx      26
        obsScales   1.0
    }
    jointVel
    {
        startIdx    0
        numIdx      26
        obsScales   0.05
    }
    jointTorque
    {
        startIdx    0
        numIdx      26
        obsScales   1.0
    }
    bodyAngVel
    {
        startIdx    0
        numIdx      3
        obsScales   1.0
    }
    baseEuler
    {
        startIdx    0
        numIdx      2
        obsScales   1.0
    }
    bodyLineAcc
    {
        startIdx    0
        numIdx      3
        obsScales   0.5
    }
}
```
- 该文件中定义了不同输入数据，包括`commandPhase、command、jointPos、jointVel、jointTorque、bodyAngVel、baseEuler、bodyLineAcc`等，可以根据实际情况进行选择，坐标轴按照`右手系`定义，其相关的数据顺序按照`xyz`定义。从上到下的文本顺序就是数据输入的排列顺序，可以根据实际情况进行修改，需要设置数据的起始维度和维度大小，以及obsScales参数，使其与训练时的输入一致。
## 手柄控制
### 注意事项
- 1.遥控器型号通过运行时launch参数，joystick_type指定。
- 2.在`src/humanoid-control/humanoid_controllers/launch/joy`目录指定了按键映射关系，新增遥控器类型可以直接添加自己的按键映射关系到json文件中，运行时通过`joystick_type:=bt2pro`传递相应文件名即可
- 3.在`src/humanoid-control/humanoid_controllers/launch/joy/joy_control_bt.launch`中的`joystick_sensitivity`参数可以调节遥控器的灵敏度，默认值为0.5。
- 4.在仿真中运行时，可以将参数设置成`joystick_type:=sim`，此时手柄控制会失效，需要通过键盘控制,终端会有相应的提示出现。
### 北通手柄控制说明
- 参考的遥控器键位如下，其他型号需要自行修改遥控器节点：
   - 该遥控器沿用了kuavo-ros-control的遥控器控制，但有些按钮和功能有所不同：
      - `A`: STANCE 同时会上锁
      - `B`: TROT / STANCE 切换
      - `X`: INTO RL-Control
      - `Y`: NONE
      - `LB`: 解锁，解锁后可以进行踏步或者其他动作

   - 摇杆控制腿部运动
      - 左摇杆控制前后左右
      - 右摇杆控制左右转
   - `start`键实物控制时用于从悬挂准备阶段切换到站立
   - `back`键用于退出所有节点

### H12遥控器控制说明
- 参考的遥控器键位如下，其他型号需要自行修改遥控器节点：
   - 该遥控器沿用了kuavo-ros-control的遥控器控制，但有些按钮和功能有所不同：
      - `D`: STANCE 同时会上锁
      - `B`: INTO RL-Control
      - `A`: NONE
      - `C`: 解锁，解锁后可以进行踏步或者其他动作

   - 摇杆控制腿部运动
      - 左摇杆控制前后左右
      - 右摇杆控制左右转
   - `F`键实物控制时用于从悬挂准备阶段切换到站立
   - `E`键用于退出所有节点

- 启动说明
  - 实物启动 
    - 启动前确保`E`建在最左边，`F`建在最右边。
    - 机器人缩腿后`F`键拨到最左边是站立
    - `E`键拨到最右边是停止程序（注意此时机器人不会自动下蹲）
