# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .custom.humanoid_config import XBotLCfg, XBotLCfgPPO
from .custom.humanoid_env import XBotLFreeEnv
from .custom.kuavo_config import KuavoCfg, KuavoCfgPPO
from .custom.kuavo_env import KuavoFreeEnv
from .custom.kuavo_s40_config import (
    KuavoS40Cfg, KuavoS40C0Cfg, KuavoS40CfgPPO,
    KuavoS40AmassCfg, KuavoS40AmassCfgPPO,
    KuavoS40SKCfg, KuavoS40SKCfgPPO,
    KuavoS40TrotCfg, KuavoS40TrotCfgPPO,
    KuavoS40LSCfg, KuavoS40LSCfgPPO,
)
from .custom.kuavo_s42_config import (
    KuavoS42Cfg, KuavoS42CfgPPO, KuavoS42LSCfg, KuavoS42LSCfgPPO, KuavoS42SKCfg, KuavoS42SKCfgPPO,
)

from .custom.kuavo_s40_env import KuavoS40FreeEnv, KuavoS40AmassFreeEnv, KuavoS40SKFreeEnv
from .custom.kuavo_s42_env import KuavoS42FreeEnv

from humanoid.utils.task_registry import task_registry


task_registry.register("humanoid_ppo", XBotLFreeEnv, XBotLCfg(), XBotLCfgPPO() )
task_registry.register("kuavo_ppo", KuavoFreeEnv, KuavoCfg(), KuavoCfgPPO() )

task_registry.register("kuavo_s40_ppo", KuavoS40FreeEnv, KuavoS40Cfg(), KuavoS40CfgPPO() )
task_registry.register("kuavo_s40_c0_ppo", KuavoS40FreeEnv, KuavoS40C0Cfg(), KuavoS40CfgPPO() )

# task_registry.register("kuavo_s40_original_ppo", KuavoS40OriginalFreeEnv, KuavoS40OriginalCfg(), KuavoS40OriginalCfgPPO() )
task_registry.register("kuavo_s40_amass_ppo", KuavoS40AmassFreeEnv, KuavoS40AmassCfg(), KuavoS40AmassCfgPPO() )
task_registry.register("kuavo_s40_sk_ppo", KuavoS40SKFreeEnv, KuavoS40SKCfg(), KuavoS40SKCfgPPO() )
task_registry.register("kuavo_s40_trot_ppo", KuavoS40FreeEnv, KuavoS40TrotCfg(), KuavoS40TrotCfgPPO() )
task_registry.register("kuavo_s40_ls_ppo", KuavoS40FreeEnv, KuavoS40LSCfg(), KuavoS40LSCfgPPO() )

task_registry.register("kuavo_s42_ppo", KuavoS42FreeEnv, KuavoS42Cfg(), KuavoS42CfgPPO() )
task_registry.register("kuavo_s42_ls_ppo", KuavoS42FreeEnv, KuavoS42LSCfg(), KuavoS42LSCfgPPO() )
task_registry.register("kuavo_s42_sk_ppo", KuavoS42FreeEnv, KuavoS42SKCfg(), KuavoS42SKCfgPPO() )
