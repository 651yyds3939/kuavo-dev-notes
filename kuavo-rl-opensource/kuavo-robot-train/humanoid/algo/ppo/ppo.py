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

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from .actor_critic import ActorCritic
from .long_short_ac import LongShortActorCritic
from .rollout_storage import RolloutStorage



def get_symm_obs(obs_batch, frame_stack=15):
    # is_standing = obs_batch.reshape(batch_size, 15, 92)[:, -1, 5].reshape(batch_size)

    batch_size = obs_batch.size(0)
    obs_hist = obs_batch.clone().reshape(batch_size, frame_stack, 92)

    def symm_func(obs_hist, start):
        obs_hist[:, :, start:start+12] = torch.roll(obs_hist[:, :, start:start+12], shifts=6, dims=2)
        obs_hist[:, :, [start, start+1, start+5, start+6, start+7, start+11]] *= -1
        obs_hist[:, :, start+12:start+26] = torch.roll(obs_hist[:, :, start+12:start+26], shifts=7, dims=2)
        obs_hist[:, :, [start+13, start+14, start+16, start+18, start+20, start+21, start+23, start+25]] *= -1

    obs_hist[:, :, [0, 1, 3, 4]] *= -1
    symm_func(obs_hist, 6)
    symm_func(obs_hist, 32)
    symm_func(obs_hist, 58)
    obs_hist[:, :, [84, 86, 87, 90]] *= -1

    return obs_hist.reshape(batch_size, -1)

def get_symm_action(action_batch):
    res = action_batch.clone()
    start = 0
    res[:, start:start + 12] = torch.roll(res[:, start:start + 12], shifts=6, dims=1)
    res[:, [start, start + 1, start + 5, start + 6, start + 7, start + 11]] *= -1
    res[:, start + 12:start + 26] = torch.roll(res[:, start + 12:start + 26], shifts=7, dims=1)
    res[:, [start + 13, start + 14, start + 16, start + 18, start + 20, start + 21, start + 23, start + 25]] *= -1
    return res

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 frame_stack=15,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.frame_stack = frame_stack

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.actor.parameters(), 'lr': learning_rate, 'weight_decay': 1e-5},
            {'params': self.actor_critic.critic.parameters(), 'lr': learning_rate},
            {'params': [self.actor_critic.std], 'lr': learning_rate},
        ])
        self.transition = RolloutStorage.Transition()

        # self.teacher = deepcopy(actor_critic)
        # self.teacher.load_state_dict(torch.load("/home/thl/Projects/humanoid-gym/logs/Kuavo_s42_ls_ppo/Dec30_15-05-12_v1/model_300.pt")["model_state_dict"])

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.frame_stack, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl_div = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        mean_kl_div += kl_mean.item()

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # dof_pos = obs_batch.reshape(batch_size, 15, 92)[:, -1, 6:32]
                # dof_vel = obs_batch.reshape(batch_size, 15, 92)[:, -1, 32:58]
                # p_gains = torch.tensor([60., 60., 60., 60., 15., 15] * 2 + [5] * 14, device=self.device)
                # d_gains = torch.tensor([0.5] * 12 + [3] * 14, device=self.device)
                # default_dof_pos = torch.tensor([
                #                         0., 0., -0.39, 0.74, -0.35, 0.,  # left joint pos
                #                         0., 0., -0.39, 0.74, -0.35, 0.  # right joint pos
                #                     ] + [0] * 14, device=self.device)
                # actions_scaled = mu_batch * 0.25
                # torques = p_gains * (actions_scaled + default_dof_pos - dof_pos) - d_gains * dof_vel
                # weight = torch.tensor([0, 0, 0, 0, 3, 5] * 2 + [0] * 14, device=self.device)
                # action_loss = torch.mean((torques * weight) ** 2)


                # # is_standing = obs_batch.reshape(batch_size, 15, 93)[:, -1, 5].reshape(batch_size)
                # # target = mu_batch.clone()
                # # target[:, :12] = torch.roll(target[:, :12], shifts=6, dims=1)
                # # target[:, 12:] = torch.roll(target[:, 12:], shifts=7, dims=1)
                # # target[:, [0, 1, 5, 6, 7, 11]] *= -1
                # # target[:, [13, 14, 16, 18, 20, 21, 23, 25]] *= -1
                # # symm_loss = torch.mean((mu_batch - target) ** 2, dim=1)
                # #

                action_loss = torch.mean((mu_batch[:, [4, 5, 10, 11]] * torch.tensor([1, 5, 1, 5], device=mu_batch.device)) ** 2)


                # l = mu_batch[:, 12:].square().mean()
                # print(l)

                symm_act = get_symm_action(self.actor_critic.act_inference(get_symm_obs(obs_batch, self.frame_stack)))
                symm_loss = (symm_act - mu_batch).pow(2).mean()

                # is_marking = (obs_batch.reshape(-1, self.frame_stack, 92)[:, -1, [2, 5]] == 0).all(dim=-1)
                # with torch.inference_mode():
                #     dagger_action = self.teacher.act_inference(obs_batch[is_marking])
                # dagger_loss = (dagger_action - mu_batch[is_marking]).pow(2).mean()

                # batch_size = obs_batch.size(0)
                # pred_lin_vel = self.actor_critic.vel_estimator(obs_batch)
                # base_lin_vel = critic_obs_batch.reshape(batch_size, 3, -1)[: , -1, 6 + 26 * 4: 6 + 26 * 4 + 3] / 2
                # lin_vel_error = (base_lin_vel - pred_lin_vel).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                # loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()

                loss += action_loss * 3e-4 + symm_loss
                # loss += dagger_loss * 0.1
                print(action_loss.item(), symm_loss.item())
                # loss += lin_vel_error * 0.3
                # print(action_loss.item(), lin_vel_error.item())
                # loss += 10  * symm_loss
                # print(symm_loss)
                # loss += 0.5 * l
                # loss += 0.1 * symm_loss[is_standing.to(bool)].mean()
                # torques_loss = 1e-6 * action_loss.pow(2).mean()
                # loss += torques_loss
                # print(torques_loss)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy_loss += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl_div /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_kl_div
