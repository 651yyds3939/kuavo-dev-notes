# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.actuators import DelayedPDActuatorCfg

from omni.isaac.lab.utils import configclass

from .actuator_pd import DelayedImplicitActuator
from .actuator_pd import DelayedPDActuator_S42
from .actuator_pd import DelayedPDActuator_S49
from dataclasses import MISSING
import torch


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


@configclass
class DelayedPDActuatorCfg_S42(DelayedPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedPDActuator_S42
    friction_static: float = 0
    activation_vel: float = torch.inf
    friction_dynamic: float = 0


@configclass
class DelayedPDActuatorCfg_S49(DelayedPDActuatorCfg):
    """Delayed joint-space PD for S49 URDF training (no ankle solver)."""

    class_type: type = DelayedPDActuator_S49
    friction_static: float = 0
    activation_vel: float = torch.inf
    friction_dynamic: float = 0
