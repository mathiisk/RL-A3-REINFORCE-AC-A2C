import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from PolicyNetwork import PolicyNetwork



class REINFORCEAgent:
    def __init__(self, state_size, action_size, device, config):
        self.device = device
        self.gamma = config.gamma
        self.policy_net = PolicyNetwork(state_size, action_size, config.hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr_actor)
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

        loss = (-log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def evaluate(self, eval_env, eval_episodes=10):
        self.policy_net.eval()
        returns = []
        
        for i in range(eval_episodes):
            obs, _ = eval_env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_return = 0.0
            terminated, truncated = False, False
            
            with torch.no_grad():
                while not (terminated or truncated):
                    action, _ = self.select_action(state, greedy=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action.item())
                    episode_return += reward
                    state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
            returns.append(episode_return)
        
        self.policy_net.train()
        return np.mean(returns)
            