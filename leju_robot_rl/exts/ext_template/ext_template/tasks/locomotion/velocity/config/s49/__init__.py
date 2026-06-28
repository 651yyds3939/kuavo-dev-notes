# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# S49（四代机）全身舞训练 / 回放环境
# 与 s42/ 目录并行，不影响原有 S42/S46 训练链路
##

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S49-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS49PunchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS49FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S49-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS49PunchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS49FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S49-Punch-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.punch_env_cfg:KuavoS49PunchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS49FlatPPORunnerCfg",
    },
)
