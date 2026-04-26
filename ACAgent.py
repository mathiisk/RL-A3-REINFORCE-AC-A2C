import numpy as np
import copy
import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical

from Networks import PolicyNetwork
from Networks import QValueNetwork

class ACAgent:
    def __init__(self, state_size, action_size, device, config):
        self.device = device
        self.gamma = config.gamma
        self.tau = config.tau
        self.update_very = config.target_update_freq

        self.policy_net = PolicyNetwork(state_size, action_size, config.hidden_size).to(device)
        self.value_net = QValueNetwork(state_size, action_size, config.hidden_size).to(device)

        # Target network: copy of critic, not trained directly
        self.target_net = copy.deepcopy(self.value_net)
        for p in self.target_net.parameters():
            p.requires_grad = False  # never backprop through target

        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=config.lr_critic)
        self.steps_done = 0

    def _soft_update_target(self):
        """Polyak averaging: target = tau*online + (1-tau)*target"""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )

    def select_action(self, state, greedy=False):
        probs = self.policy_net(state)
        dist = Categorical(probs)
        if greedy:
            action = probs.argmax(dim=-1)
            return action, None
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def update(self, states, actions, rewards, next_states, next_actions, dones):
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # ── Critic loss ──────────────────────────────────────────────
        q_values = self.value_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            # Use TARGET network for bootstrap — this is the key stability fix
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
        target = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)
        critic_loss = torch.nn.functional.mse_loss(q_values, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor loss ───────────────────────────────────────────────
        # Separate backward pass so critic grads aren't contaminated
        probs = self.policy_net(states)
        log_probs = Categorical(probs).log_prob(actions)
        actor_loss = (-log_probs * q_values.detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.steps_done % 100 == 0:
            self._soft_update_target()

    # evaluate() is unchanged
    def evaluate(self, eval_env, eval_episodes=10):
        self.policy_net.eval()
        returns = []
        for i in range(eval_episodes):
            result = eval_env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_return = 0.0
            terminated, truncated = False, False
            with torch.no_grad():
                while not (terminated or truncated):
                    action, _ = self.select_action(state, greedy=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action.item())
                    episode_return += reward
                    state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            returns.append(episode_return)
        self.policy_net.train()
        return np.mean(returns)

def train_AC(params, device):
    env = gym.vector.SyncVectorEnv([
        lambda: gym.make("CartPole-v1") for _ in range(params.num_envs)
    ])
    eval_env = gym.make("CartPole-v1")
    n_actions = env.single_action_space.n
    n_observations = env.single_observation_space.shape[0]

    agent = ACAgent(n_observations, n_actions, device, params)
    eval_returns = []

    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    states = torch.tensor(obs, dtype=torch.float32, device=device)
    actions, _ = agent.select_action(states)

    while agent.steps_done < params.total_steps:

        obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
        dones = terminated | truncated
        next_states = torch.tensor(obs, dtype=torch.float32, device=device)
        next_actions, _ = agent.select_action(next_states)
        agent.update(states, actions, rewards, next_states, next_actions, dones)

        agent.steps_done += params.num_envs
        states = next_states
        actions = next_actions

        if agent.steps_done % params.evaluate_every < params.num_envs:
            ret = agent.evaluate(eval_env, eval_episodes=params.eval_episodes)
            eval_returns.append(ret)
            print(f'Steps: {agent.steps_done} | Reward: {ret:.1f}')

    env.close()
    eval_env.close()
    return eval_returns