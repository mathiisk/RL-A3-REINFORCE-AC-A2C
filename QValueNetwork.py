import torch.nn as nn

class QValueNetwork(nn.Module):
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
        return self.net(x).squeeze(-1)