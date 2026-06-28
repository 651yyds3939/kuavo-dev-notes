# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
# =========================================================================
# 1. 训练环境注册：强行将其拦截并重定向到我们的打拳配置
# =========================================================================
gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 🟢 核心修改：将原来的 flat_env_cfg:KuavoS42FlatEnvCfg 改为 punch_env_cfg
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS42PunchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

# =========================================================================
# 2. 回放/评估环境注册：同样重定向到打拳的 PLAY 配置
# =========================================================================
gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 🟢 核心修改：改为播放模式专用的类 KuavoS42PunchEnvCfg_PLAY
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS42PunchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-DreamWaq-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-DreamWaq-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-DreamWaq-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-DreamWaq-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-Punch-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 完美的解耦调用：精准指向你刚新建的 punch_env_cfg.py 和里面的 KuavoS42PunchEnvCfg 类
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS42PunchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# TD-MPC2 world-model training tasks (S42/S46 stable asset)
# ---------------------------------------------------------------------------
gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Punch-ArmsOnly-TDMPC2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS42PunchArmsOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
        "tdmpc2_cfg_entry_point": f"{agents.__name__}.tdmpc2_cfg:KuavoS42ArmsOnlyTDMPC2RunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Punch-ArmsOnly-TDMPC2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS42PunchArmsOnlyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
        "tdmpc2_cfg_entry_point": f"{agents.__name__}.tdmpc2_cfg:KuavoS42TDMPC2PlayRunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Punch-TDMPC2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS42PunchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
        "tdmpc2_cfg_entry_point": f"{agents.__name__}.tdmpc2_cfg:KuavoS42DanceTDMPC2RunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Velocity-TDMPC2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tdmpc2_velocity_env_cfg:KuavoS42VelocityTDMPC2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
        "tdmpc2_cfg_entry_point": f"{agents.__name__}.tdmpc2_cfg:KuavoS42VelocityTDMPC2RunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Punch-TDMPC2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS42PunchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
        "tdmpc2_cfg_entry_point": f"{agents.__name__}.tdmpc2_cfg:KuavoS42TDMPC2PlayRunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Velocity-TDMPC2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tdmpc2_velocity_env_cfg:KuavoS42VelocityTDMPC2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
        "tdmpc2_cfg_entry_point": f"{agents.__name__}.tdmpc2_cfg:KuavoS42VelocityTDMPC2PlayRunnerCfg",
    },
)
