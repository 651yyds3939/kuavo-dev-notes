# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.actuators import DelayedPDActuator
from omni.isaac.lab.utils import DelayBuffer

from .ankle_s42 import joint_to_motor_position, get_joint_dumping_torque

if TYPE_CHECKING:
    from .actuator_cfg import DelayedImplicitActuatorCfg
    from .actuator_cfg import DelayedPDActuatorCfg_S42
    from .actuator_cfg import DelayedPDActuatorCfg_S49


class DelayedImplicitActuator(ImplicitActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        self.velocities_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        self.efforts_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        # all of the envs
        self._ALL_INDICES = torch.arange(
            self._num_envs, dtype=torch.long, device=self._device
        )

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(
            control_action.joint_positions
        )
        control_action.joint_velocities = self.velocities_delay_buffer.compute(
            control_action.joint_velocities
        )
        control_action.joint_efforts = self.efforts_delay_buffer.compute(
            control_action.joint_efforts
        )
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


class DelayedPDActuator_S42(DelayedPDActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedPDActuatorCfg_S42
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedPDActuatorCfg_S42, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.)

        self.gym2lab = [0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25]
        self.lab2gym = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25]
        self.dof_pos_illegal = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self.dof_pos_illegal[env_ids] = 0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        motor_pos = joint_to_motor_position(joint_pos[:, self.lab2gym])[:, self.gym2lab]
        # check
        self.dof_pos_illegal |= torch.isnan(motor_pos).any(dim=-1)
        motor_pos[self.dof_pos_illegal] = 0

        damping_torque = - get_joint_dumping_torque(joint_pos[:, self.lab2gym], motor_pos[:, self.lab2gym], self.damping[:, self.lab2gym], joint_vel[:, self.lab2gym])[:, self.gym2lab]

        # check
        self.dof_pos_illegal |= torch.isnan(damping_torque).any(dim=-1)
        damping_torque[self.dof_pos_illegal] = 0

        # calculate the desired joint torques
        # CSP
        self.computed_effort = self.stiffness * error_pos + damping_torque + control_action.joint_efforts \
            - (self.friction_static * torch.tanh(joint_vel / self.activation_vel) - self.friction_dynamic * joint_vel)
        # CST
        # self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts \
        #     - (self.friction_static * torch.tanh(joint_vel / self.activation_vel) + self.friction_dynamic * joint_vel)
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class DelayedPDActuator_S49(DelayedPDActuator):
    """Joint-space delayed PD for S49 URDF training (no S42 ankle parallel solver).

    The S42 ankle solver + lab2gym mapping is calibrated for biped_s46.usd. Using it on
    runtime-imported S49 URDF causes NaN torques and dof_pos_illegal terminations.
    """

    cfg: DelayedPDActuatorCfg_S49

    def __init__(self, cfg: DelayedPDActuatorCfg_S49, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.0)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.0)
        self.dof_pos_illegal = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self.dof_pos_illegal[env_ids] = 0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
            - (
                self.friction_static * torch.tanh(joint_vel / self.activation_vel)
                - self.friction_dynamic * joint_vel
            )
        )
        self.dof_pos_illegal |= torch.isnan(self.computed_effort).any(dim=-1)
        self.computed_effort[self.dof_pos_illegal] = 0.0

        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
