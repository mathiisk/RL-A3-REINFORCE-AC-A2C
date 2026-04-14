import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)