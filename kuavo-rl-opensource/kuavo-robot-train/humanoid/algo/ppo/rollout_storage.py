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

def reverse_cumsum(x, dim=1):
    return x + x.sum(dim=dim, keepdim=True) - x.cumsum(dim=dim)

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
        
        def clear(self):
            self.__init__()

    def __init__(
            self,
            num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape,
            frame_stack=100,
            device='cpu',
            save_memory=True,
    ):

        self.device = device
        self.num_envs = num_envs

        self.save_memory = save_memory
        self.frame_stack = frame_stack
        if len(obs_shape) == 1 and obs_shape[0] % frame_stack == 0:
            self.single_obs_dim = obs_shape[0] // frame_stack
        else:
            self.single_obs_dim = None

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        if not self.save_memory:
            self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        else:
            assert self.single_obs_dim is not None, "Frame stack is only supported for single observation dimension"
            self.observations = torch.zeros(
                num_envs, num_transitions_per_env + frame_stack - 1, self.single_obs_dim,
                device=self.device
            )

        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0
        self.first_step = True

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        if self.save_memory:
            obs = transition.observations.reshape(self.num_envs, self.frame_stack, self.single_obs_dim)
            if self.first_step:
                self.observations[:, :self.frame_stack].copy_(obs)
                self.first_step = False
            else:
                self.observations[:, self.frame_stack + self.step - 1].copy_(obs[:, -1])
        else:
            self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0
        self.first_step = True

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        padded_dones = torch.cat([
            torch.zeros(self.num_envs, self.frame_stack - 1, device=self.device),
            self.dones[:, :, 0].transpose(0, 1)
        ], dim=1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                if self.save_memory:
                    step_idx, env_idx = batch_idx // self.num_envs, batch_idx % self.num_envs
                    step_indices = step_idx.unsqueeze(1) + torch.arange(self.frame_stack).unsqueeze(0).to(step_idx.device)
                    env_indices = env_idx.unsqueeze(1).expand(-1, self.frame_stack)
                    obs_batch = self.observations[env_indices, step_indices]

                    dones = padded_dones[env_indices, step_indices]
                    dones_mask = torch.zeros_like(dones)
                    dones_mask[:, :-1] = reverse_cumsum(dones[:, :-1], dim=1)
                    dones_mask = dones_mask.to(torch.bool).unsqueeze(-1)

                    obs_batch.masked_fill_(dones_mask, 0)
                    obs_batch = obs_batch.reshape(mini_batch_size, -1)
                else:
                    obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None


if __name__ == '__main__':
    num_envs = 1024
    num_transitions_per_env = 10
    single_obs_dim = 92
    privileged_obs_shape = (None,)
    actions_shape = (26,)
    frame_stack = 15
    obs_shape = (single_obs_dim * frame_stack,)

    rollout_storage = RolloutStorage(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, frame_stack)
    obs_buf = torch.randn(num_envs, frame_stack, single_obs_dim)

    for i in range(num_transitions_per_env):
        transition = RolloutStorage.Transition()
        if i != 0:
            new_obs = torch.randn(num_envs, single_obs_dim)
            obs_buf = torch.roll(obs_buf, shifts=-1, dims=1)
            obs_buf[:, -1] = new_obs.clone()
        transition.observations = obs_buf.reshape(num_envs, -1).clone()
        transition.actions = torch.randn(num_envs, *actions_shape)
        transition.rewards = torch.randn(num_envs, )
        transition.dones = torch.randint(0, 2, (num_envs, )).byte()
        transition.values = torch.randn(num_envs, 1)
        transition.actions_log_prob = torch.randn(num_envs, )
        transition.action_mean = torch.randn(num_envs, *actions_shape)
        transition.action_sigma = torch.randn(num_envs, *actions_shape)

        rollout_storage.add_transitions(transition)

    for i in rollout_storage.mini_batch_generator(4):
        pass
