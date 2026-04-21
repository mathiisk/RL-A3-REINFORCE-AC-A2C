import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical

from Nets import PolicyNetwork
from nets import ValueNetwork

class A2CAgent_Bootstrap():
    def __init__(self, state_size, action_size, device, config):
        self.device = device
        self.gamma = config.gamma
        self.policy_net = PolicyNetwork(state_size, action_size, config.hidden_size).to(device)
        self.value_net = ValueNetwork(state_size, config.hidden_size).to(device)

        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=config.lr_critic)
        self.steps_done = 0


    def select_action(self, state, greedy=False):
        probs = self.policy_net(state)
        dist = Categorical(probs)

        if greedy:
            action = probs.argmax(dim=-1)
            return action, None

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def update(self, states, actions, rewards):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        std = returns.std()
        if std > 1e-8:
            returns = (returns - returns.mean()) / std
        else:
            returns = returns - returns.mean()

        # recompute log_probs fresh from stored states/actions
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        probs = self.policy_net(states_tensor)
        log_probs = Categorical(probs).log_prob(actions_tensor)

        state_value = self.value_net(states_tensor)
        advantage = returns - state_value
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = ((advantage) ** 2).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

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


def train_A2C_Bootstrap(params, device):
    env = gym.vector.SyncVectorEnv([
        lambda: gym.make("CartPole-v1") for _ in range(params.num_envs)
    ])
    eval_env = gym.make("CartPole-v1")
    n_actions = env.single_action_space.n
    n_observations = env.single_observation_space.shape[0]

    agent = A2CAgent_Bootstrap(n_observations, n_actions, device, params)
    eval_returns = []

    states_buf  = [[] for _ in range(params.num_envs)]
    actions_buf = [[] for _ in range(params.num_envs)]
    rewards_buf = [[] for _ in range(params.num_envs)]

    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    states = torch.tensor(obs, dtype=torch.float32, device=device)

    while agent.steps_done < params.total_steps:
        actions, _ = agent.select_action(states)
        obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
        dones = terminated | truncated


        for i in range(params.num_envs):
            states_buf[i].append(states[i])
            actions_buf[i].append(actions[i])
            rewards_buf[i].append(rewards[i])

        for i in range(params.num_envs):
            if dones[i]:
                agent.update(states_buf[i], actions_buf[i], rewards_buf[i])
                states_buf[i]  = []
                actions_buf[i] = []
                rewards_buf[i] = []

        agent.steps_done += params.num_envs
        states = torch.tensor(obs, dtype=torch.float32, device=device)

        if agent.steps_done % params.evaluate_every < params.num_envs:
            ret = agent.evaluate(eval_env, eval_episodes=params.eval_episodes)
            eval_returns.append(ret)
            print(f'Steps: {agent.steps_done} | Reward: {ret:.1f}')

    env.close()
    eval_env.close()
    return eval_returns