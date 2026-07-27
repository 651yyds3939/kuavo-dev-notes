# Sim2Sim 纯 RL 部署问题 — AI 修复任务

## 背景

我们在做人形机器人 Kuavo S49（26 自由度：12 腿 + 14 臂）的原地舞蹈强化学习。技术栈：Isaac Lab 1.4.1 + Isaac Sim 4.2.0 + PPO (rsl_rl) 训练，MuJoCo 3.0.1 仿真验证。

**当前状态**：经过 18 版奖励函数迭代，v18（`2026-07-17_05-51-49`）在 Isaac 训练中表现优秀——机器人站稳、手臂流畅跟踪 CSV 舞蹈轨迹。但部署到 MuJoCo 纯 RL 模式（`rl_from_start:=true sim_keep_wbc_with_rl:=false`）时，机器人出生后几秒内必然摔倒。

## 核心症状

- **Isaac 训练**：机器人稳定站立，手臂跟踪准确，动作流畅
- **MuJoCo 部署**：机器人出生后向前倾倒（不是劈叉/扭曲），几秒内倒地
- **bag 数据验证**：首帧观测值已完全对齐（`gravity_body z=-0.999`, `jointPos` 膝=0.52, `action max=0.56`），网络输入正确、输出未饱和
- **问题定位**：Sim2Sim 观测/动作链路已全部对齐（修复了 8 处 Bug），残余问题是 Isaac PhysX 和 MuJoCo 两个物理引擎在相同 PD 目标下产生不同的动力学行为

## 关键文件路径

### 训练侧
```
/home/lwy/Notes/kuavo-dev-notes/leju_robot_rl/
├── scripts/rsl_rl/play.py                              ← ONNX 导出（观测/动作重排）
├── exts/ext_template/ext_template/tasks/locomotion/velocity/config/s49/punch_env_cfg.py  ← 奖励/观测配置
├── exts/ext_template/ext_template/tasks/locomotion/velocity/mdp/rewards.py                ← 奖励函数实现
├── kuavo_action_S49_FROM_S54_INPLACE_RAD.csv           ← 1565帧舞蹈CSV（26列弧度值）
└── logs/rsl_rl/Kuavo/s49/dance/2026-07-17_05-51-49/    ← v18训练产物
    ├── RUN_CONFIG.md                                    ← 训练结果+最终奖励值
    ├── params/env.yaml                                  ← 完整配置快照
    └── exported/policy_s42.onnx                         ← 部署用ONNX（115维输入/26维输出）
```

### 部署侧
```
~/kuavo_all/kuavo-rl-opensource/kuavo-robot-deploy/
├── src/humanoid-control/humanoid_controllers/
│   ├── src/humanoidController.cpp    ← 控制器（观测组装、动作下发、PD控制）
│   ├── config/kuavo_v49/rl/skw_rl_param_dance.info  ← 115维观测配置
│   └── model/networks/49dance_v18.onnx               ← 部署用ONNX
├── src/mujoco/src/mujoco_node.cc     ← MuJoCo物理节点（传感器映射、PID控制）
├── analyze_r_takeover_bag.py         ← bag分析脚本（解析115维观测各块）
└── ...
```

## 已完成的对齐修复（8 处 Bug）

### play.py（2 处）

**Bug 1**：S46 的 gym2lab/lab2gym 映射被错误用于 S49。S49 的 mujoco_node 用 Lab 序（`leg_l1..l6, leg_r1..r6, zarm_.*`），ONNX 用 S46 映射重排了 S49 数据 → 关节全错位。修复：改为恒等映射 `list(range(26))`。

**Bug 2**：referenceJointPos（offset 61）也被 gym2lab 重排，但 C++ 端已是 Lab 序 → 二次重排搅碎参考值。修复：只对 action（offset 87）做重排。

### humanoidController.cpp（5 处）

**Bug 3**：观测 jointPos 基准不一致。训练 `joint_pos_rel = sensor - 0`（URDF 零位，膝=0.52）。C++ `jointPos = sensor - defalutJointPos_`（站姿基准，膝=0）。修复：`sim_joint_obs_offset_ = -defalutJointPos_`，使 `jointPos = sensor`。

**Bug 4**：referenceJointPos 基准不一致。C++ 的 CSV 加载用 `CSV - defalutJointPos_`，训练用 `CSV - 0`。修复：rl_from_start 时 `referenceJointPos_ += defalutJointPos_`。

**Bug 5a**：action→q_des 基准不一致。训练 `q_des = action × 0.30 + 0`。C++ 额外加了 `defalutJointPos_`（膝 +0.52）。修复：rl_from_start 时只加 0。

**Bug 5b**：settle 阶段（RL 推理前）action=0，基准为 0 → PD 把膝从 0.52 推向 0。修复：settle 时保留 `defalutJointPos_` 基准。

**Bug 6**：sagittal 腿关节轴符号翻转。Isaac USD 转换时 6 个关节（髋俯仰/膝/踝，左右各 3）的轴与 MuJoCo URDF 反向。膝=-0.52 vs 训练 +0.52。修复：观测和动作两端对索引 [2,3,4,8,9,10] 做符号翻转。

### mujoco_node.cc（2 处）

**Bug 7**：首帧命令到达前 `pending_joint_cmd` 为空 → PD 回退 `q_des = q`（当前位置）→ 零力矩自由落体。修复：回退到 `qpos_init[qa]`（spawn 站姿）。

**Bug 8**：首帧 `use_rl_native_pd` 为 false → 走老 PID 路径用顺序索引 `qpos[7+i]` 读关节，S49 的 qpos 混了手指 → 读到错误关节。修复：qpos_init 加载后预填充 `pending_joint_cmd`，`use_rl_native_pd=true`。

## 已验证的成果

修复后用 `analyze_r_takeover_bag.py` 分析 bag 首帧数据：

```
gravity_body:      [0.04, -0.02, -0.999]    ← 几乎完美竖直
jointPos[3] (膝):   0.519                    ← 正确站姿（训练值=0.52）
jointPos[4] (踝):  -0.212                    ← 接近站姿（训练值=-0.30）
referenceJointPos:  [0, 0, -0.27, 0.52, ...] ← 匹配训练
action max:         0.56                     ← 未饱和（坏的情况是 ±8.0）
```

**观测链路已完全对齐。网络看到正确的输入，输出正常的动作。但机器人仍向前倾倒。**

## 待解决的问题

### 根本原因

Isaac 和 MuJoCo 使用不同的物理引擎（PhysX vs MuJoCo），相同 PD 目标产生不同的动力学行为。具体可能包括：

1. **接触模型差异**：PhysX 和 MuJoCo 的接触刚度、阻尼、摩擦模型不同。脚步与地面的接触力在 MuJoCo 中可能偏小，导致机器人站不稳。

2. **PD 增益在不同引擎中的等效性**：训练中 Isaac 的 `DelayedPDActuator_S49` 执行器包含力矩限制和延迟，MuJoCo 的 PD 直接用 Kp/Kd。Kp=150/100/40 在 MuJoCo 中可能不够刚。

3. **关节阻尼/摩擦力建模差异**：Isaac 的 joint friction 和 MuJoCo 的 joint damping 参数可能不同。

4. **质量/惯量分布的微小差异**：训练用 lite URDF（`biped_s49_26dof_lite.urdf`），MuJoCo 用完整 URDF（`biped_s49.urdf`）。虽然总质量相同（58.4kg），但链节质量和惯量分布可能不同。

5. **域随机化覆盖范围**：训练中的域随机化（质量 ±5%、摩擦 0-2、执行器增益 ±20%）可能没有覆盖 MuJoCo 的实际物理参数范围。

### 你的任务

让 Kuavo S49 在 MuJoCo 纯 RL 模式下不摔倒，行为尽可能接近 Isaac 训练中的表现。

**方式一：调 MuJoCo 侧的 PD 增益**

文件：`skw_rl_param_dance.info` 中的 `jointKp`/`jointKd`。
当前值：髋 100、膝 150、踝 40。Isaac 训练侧力矩上限被放宽（髋 180/膝 100 vs 真实 127/135），MuJoCo 侧 Kp 可能不够。尝试大幅提高 Kp（比如 300/200/100）看能否撑住站姿。

**方式二：在 MuJoCo 侧对齐物理参数**

文件：MuJoCo XML 模型 `biped_s49/xml/scene_rl.xml` 或 URDF。
检查接触刚度、阻尼、关节摩擦等参数是否与 Isaac PhysX 设定一致。

**方式三：训练侧域随机化覆盖 MuJoCo 物理**

修改 `punch_env_cfg.py` 的 `EventCfg`，扩大域随机化范围（质量 ±20%、PD 增益 0.5-1.5×），使训练策略对 MuJoCo 的物理参数更鲁棒，然后重训。

**方式四：在 MuJoCo 中做少量微调（fine-tune）**

用 Isaac 训练的 v18 checkpoint 作为初始化，在 MuJoCo 环境中做少量 PPO 迭代（类似 domain randomization finetune）。

**方式五：接受物理引擎差异，走 Hybrid WBC+RL 路线**

用 `sim_keep_wbc_with_rl:=true`，让 WBC 负责站姿稳定，RL 只控制手臂。观测链路已对齐，手臂表现应显著改善。这是当前最快上真机的路径。

## 调试工具

**bag 分析**：部署仓根目录的 `analyze_r_takeover_bag.py`
```bash
# MuJoCo 启动后录 bag
rosbag record -O debug.bag /rl_controller/singleInputData /rl_controller/actions
# 分析首帧观测
python3 analyze_r_takeover_bag.py debug.bag
```

**健康首帧判据**：
| 指标 | 健康 | 异常 |
|------|------|------|
| gravity_body[2] | ≈ -1.0 | > -0.95 |
| jointPos[3] (膝) | ≈ +0.52 | 负值 |
| action max | < 3.0 | > 7.0 |
| jointVel max | < 2.0 | > 10 |

## 训练命令

```bash
cd ~/kuavo_all/leju_robot_rl && conda activate isaaclab
bash scripts/tools/run_s49_lafan1_retrain.sh
```

## MuJoCo 启动命令

```bash
# 容器内
cd /root/kuavo_ws
source devel/setup.zsh
export ROBOT_VERSION=49

# 纯 RL 模式（当前有问题的模式）
roslaunch humanoid_controllers load_kuavo_mujoco_sim_dance_s49.launch \
  joystick_type:=sim sim_keep_wbc_with_rl:=false rl_from_start:=true

# Hybrid 模式（当前可用的模式）
roslaunch humanoid_controllers load_kuavo_mujoco_sim_dance_s49.launch joystick_type:=sim
```

## 项目文档

```
/home/lwy/Notes/kuavo-dev-notes/kuavo_notes/
├── 23.1.RL_dance_terminal_commands.md        ← 终端命令全集
├── 23.2.RL_dance_overview.md                 ← 项目总览
├── 23.4.RL_dance_train.md                    ← 训练架构
├── 23.5.RL_dance_reward_iterate.md           ← 奖励迭代历史
├── 23.6.RL_dance_sim2sim.md                  ← Sim2Sim 基础知识
├── 23.7.RL_dance_deploy_hybrid.md            ← Hybrid WBC+RL 部署
└── 23.8.RL_dance_pure_rl_sim2sim_debug.md    ← 8 处 Bug 修复记录
```
