# leju_robot_rl — 代码协作与 RL 流水线

> **本目录说明**：本仓为 [`kuavo-dev-notes`](../README.md) 子目录。S49 舞蹈见 [`kuavo_notes/23.*`](../kuavo_notes/23.1.RL_dance_overview.md)；安装与命令见 [`README.md`](./README.md)。

本文档归纳强化学习的**逻辑关系**、**从零训练**含义，以及各文件按**执行时间线**的协作。仓库基于 NVIDIA Isaac Lab + RSL-RL PPO。

---

本文档归纳本仓库中人形机器人强化学习的**逻辑关系**、**从零训练**的含义，以及各代码文件如何按**执行时间线**协作。仓库根目录另有 `README.md`，侧重安装与运行命令。

---

## 一、整体逻辑：仿真 MDP + PPO

本项目基于 **NVIDIA Isaac Lab**：

- **环境**：`ManagerBasedRLEnv`（命令 / 观测 / 动作 / 奖励 / 终止 / 事件 / 课程分模块管理）。
- **算法**：**RSL-RL** 的 **PPO**，由 **`OnPolicyRunner`** 驱动。

数据流概括为：

1. **Commands**：例如 `base_velocity`，周期性采样期望线速度、角速度 / 航向。
2. **Actions**：策略输出关节目标（`JointPositionActionCfg`，带 scale），经执行器作用到机器人。
3. **仿真步进**：Isaac 物理 + 传感器（接触力、射线高度等）。
4. **Observations**：拼成 policy / critic 向量（Kuavo rough 配置中对 critic 附加多项）。
5. **Rewards**：每个 `RewTerm` 对应一项标量，乘 `weight` 后由 **`RewardManager`** 求和为每步总奖励。
6. **Terminations**：超时、非法接触、关节限位等结束 episode。
7. **Events**：重置位姿 / 关节、域随机化、间歇推扰等。
8. **Curriculum**：例如按跟踪奖励调节地形难度（见 `mdp/curriculums.py` 中对 reward episode 累计的读取）。

算法侧：`scripts/rsl_rl/train.py` 使用 **`RslRlVecEnvWrapper`** 封装环境后交给 **`OnPolicyRunner`**，内部为标准 **on-policy rollout → PPO 更新**。奖励只出现在环境 **`step`** 的返回值中，**PPO 代码不定义任务奖励**。

---

## 二、「从零训练」指什么

入口：`scripts/rsl_rl/train.py`。

1. **`AppLauncher`** 拉起 Isaac Sim。
2. **`import ext_template.tasks`** 完成 Gym 任务注册。
3. **`@hydra_task_config`** 根据 `--task` 载入 **`EnvCfg`** 与 **`RslRlOnPolicyRunnerCfg`**。
4. **`gym.make(args_cli.task, cfg=env_cfg)`** 创建环境。
5. **`OnPolicyRunner(env, ...)`**：默认**随机初始化**策略网络。
6. **`runner.learn(..., init_at_random_ep_len=True)`** 开始采样与更新。

仅在 agent 配置中启用 **`resume`** 并指定 checkpoint 时才会 **`runner.load(...)`** 续训；否则即为从零开始的随机策略与环境交互。

任务注册示例（Kuavo Flat / Rough）在：

`exts/ext_template/ext_template/tasks/locomotion/velocity/config/s42/__init__.py`

---

## 三、奖励相关文件（重要程度）

### 本仓库内自定义奖励公式

- **`exts/ext_template/ext_template/tasks/locomotion/velocity/mdp/rewards.py`**  
  Kuavo / 双足相关自定义项，例如：`feet_air_time_clip`、`feet_slide`、`track_lin_vel_xy_yaw_frame_exp`、`track_ang_vel_z_world_exp`、`contact_forces`、`stand_still_without_cmd`、`gravity_aligned_when_stopping`、`joint_power_l2`、`action_smoothness_l2` 等。

### 奖励「用哪些项、权重与参数」——最常改的配置

- **`exts/ext_template/ext_template/tasks/locomotion/velocity/config/s42/rough_env_cfg.py`** 中的 **`RewardsCfg`**：Rough 地形训练的完整奖励表。
- **`exts/ext_template/ext_template/tasks/locomotion/velocity/config/s42/flat_env_cfg.py`**：`KuavoS42FlatEnvCfg` **继承** `KuavoS42RoughEnvCfg`，主要改地形与观测；奖励多为少量 override。

### MDP 命名空间合并（自定义 + Isaac Lab 内置）

`exts/ext_template/ext_template/tasks/locomotion/velocity/mdp/__init__.py` 中：

```python
from omni.isaac.lab.envs.mdp import *  # 通用 MDP 项（含大量 reward）
from .rewards import *                   # 本仓库 rewards.py
```

因此许多惩罚项若未在本地 **`rewards.py`** 重写，其实现在 **`omni.isaac.lab.envs.mdp`**（随 Isaac Lab 安装），仅通过在 **`RewardsCfg`** 里配置 **`RewTerm`** 引用。

### 与奖励统计相关但非「奖励公式」本身

- **`exts/ext_template/ext_template/tasks/locomotion/velocity/mdp/curriculums.py`**：地形课程会读取 **`track_lin_vel_xy_exp` / `track_ang_vel_z_exp`** 等 episode 累计奖励以升降难度。

---

## 四、按时间线：哪个文件调用哪个类

### 阶段 0 —— import 时（尚未创建环境）

**文件：** `scripts/rsl_rl/train.py`  
执行 **`import ext_template.tasks`** 后：

1. **`ext_template/ext_template/__init__.py`** → **`from .tasks import *`**
2. **`ext_template/tasks/__init__.py`** → **`omni.isaac.lab_tasks.utils.import_packages`** 递归导入子包。
3. 执行到 **`tasks/locomotion/velocity/config/s42/__init__.py`** 时，多次调用 **`gym.register(...)`**：
   - **`entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv"`**
   - **`kwargs`**：`env_cfg_entry_point`、`rsl_rl_cfg_entry_point`

至此仅为 Gymnasium **注册**，未实例化环境。

### 阶段 1 —— `train.py` 中 `main` 与 Hydra

**装饰器：** `@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")`

根据 `--task` 加载：

- **`ManagerBasedRLEnvCfg` 子类**，例如 **`KuavoS42RoughEnvCfg`** / **`KuavoS42FlatEnvCfg`**（定义于 `rough_env_cfg.py`、`flat_env_cfg.py`）。
- **`RslRlOnPolicyRunnerCfg` 子类**，例如 **`KuavoS42RoughPPORunnerCfg`**（`config/s42/agents/rsl_rl_ppo_cfg.py`）。

CLI 可覆盖 **`num_envs`、`max_iterations`、`device`、`seed`** 等。

### 阶段 2 —— `gym.make` 创建环境（Isaac Lab 内部为主）

```python
env = gym.make(args_cli.task, cfg=env_cfg, ...)
```

- Gymnasium 根据 registry 实例化 **`ManagerBasedRLEnv`**（Isaac Lab）。
- **`env_cfg`** 为 **`KuavoS42RoughEnvCfg`** 等；其中 **`KuavoS42RoughEnvCfg`** 继承 **`LocomotionVelocityRoughEnvCfg`**（来自 **`omni.isaac.lab_tasks`**），并在本仓库 **`rough_env_cfg.py`** 中覆盖 **`scene`、`commands`、`observations`、`rewards`、`terminations`、`events`、`curriculum`**。
- **`KuavoS42RoughEnvCfg.__post_init__`** 中：**`self.scene.robot = Kuavos46_CFG.replace(...)`**，机器人配置来自 **`ext_template/assets/kuavo.py`**。

Isaac Lab 内部会根据 **`EnvCfg`** 构建场景与各 **Manager**（Observation / Reward / Termination / Command / Action / Event / Curriculum）。奖励项里的 **`func=mdp.xxx`** 指向 **`ext_template.tasks.locomotion.velocity.mdp`**。

### 阶段 3 —— RSL-RL 封装

```python
env = RslRlVecEnvWrapper(env)
```

**类：** **`omni.isaac.lab_tasks.utils.wrappers.rsl_rl.RslRlVecEnvWrapper`**（Isaac Lab）。  
将 Gymnasium 环境适配为 **`rsl_rl.env.VecEnv`** 接口。

### 阶段 4 —— `OnPolicyRunner` 构造

**文件：** `rsl_rl/rsl_rl/runners/on_policy_runner.py`

```python
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=..., device=...)
```

- **`env.get_observations()`** → 触发观测合成（配置来自 **`ObservationsCfg`**，函数多在 **`mdp/observations.py`** 或 **`omni.isaac.lab.envs.mdp`**）。
- **`eval(policy_cfg["class_name"])`** → 通常为 **`ActorCritic`**（`rsl_rl/rsl_rl/modules/actor_critic.py`）。
- **`eval(alg_cfg["class_name"])`** → **`PPO`**（`rsl_rl/rsl_rl/algorithms/ppo.py`）。
- **`init_storage`** → **`RolloutStorage`**（`rsl_rl/rsl_rl/storage/rollout_storage.py`）。

若启用 **`resume`**：**`runner.load`** 加载 checkpoint（仍在 **`train.py`** 中编排）。

### 阶段 5 —— `runner.learn` 循环（每一步）

对每个 PPO iteration，内层重复 **`num_steps_per_env`** 次：

| 顺序 | 调用关系 | 含义 |
|------|-----------|------|
| ① | **`OnPolicyRunner`** → **`self.alg.act(obs, critic_obs)`** | **`PPO`** → **`ActorCritic`** 输出 **`actions`** |
| ② | **`OnPolicyRunner`** → **`self.env.step(actions)`** | **`RslRlVecEnvWrapper`** → **`ManagerBasedRLEnv.step`** |
| ③ | Isaac Lab 一步内部 | **ActionManager**、物理 **`sim.step`**、传感器更新 |
| ④ | 同上 | **RewardManager** 对每个 **`RewTerm`** 调用 **`func`**（本仓库 **`mdp/rewards.py`** + **`omni.isaac.lab.envs.mdp`**），加权求和得 **`rewards`** |
| ⑤ | 同上 | **TerminationManager**、命令重采样、**CurriculumManager**（若启用） |
| ⑥ | **`OnPolicyRunner`** → **`self.alg.process_env_step(...)`** | **`PPO`** 写入 **`RolloutStorage`** |

Rollout 结束后 **`self.alg.update()`** 执行 PPO 更新。

---

## 五、依赖关系总览图

```
scripts/rsl_rl/train.py
  ├─ AppLauncher（Isaac Sim）
  ├─ import ext_template.tasks
  │     └─ ext_template/tasks/__init__.py → import_packages
  │           └─ config/s42/__init__.py → gym.register(... ManagerBasedRLEnv ...)
  ├─ @hydra_task_config → KuavoS42*EnvCfg + KuavoS42*PPORunnerCfg
  │     ├─ rough_env_cfg.py / flat_env_cfg.py（EnvCfg）
  │     └─ agents/rsl_rl_ppo_cfg.py（RunnerCfg）
  ├─ gym.make → ManagerBasedRLEnv(Isaac Lab) + EnvCfg
  │     └─ assets/kuavo.py（Kuavos46_CFG）
  │     └─ mdp/*.py（Reward/Obs/Event/… 中的 func）
  ├─ RslRlVecEnvWrapper(Isaac Lab)
  └─ OnPolicyRunner(rsl_rl) → ActorCritic + PPO + RolloutStorage
```

---

## 六、修改行为的实用顺序

1. 优先调整 **`rough_env_cfg.py`** 中的 **`RewardsCfg`**（权重、开关、`params`）。
2. 需要新公式时在 **`mdp/rewards.py`** 新增函数，并在 **`RewardsCfg`** 中增加对应的 **`RewTerm`**。

---

*文档内容由仓库协作说明整理而成；Isaac Lab / Omniverse 中未 vendoring 的类名与行为以实际安装版本为准。*
