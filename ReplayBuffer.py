import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity, obs_dim=4, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.index = 0 # pointer where to insert new transition
        self.size = 0  # size of the filled transitions

        self.states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity,),         dtype=np.int64)
        self.rewards     = np.zeros((capacity,),         dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones       = np.zeros((capacity,),         dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.index]      = state
        self.actions[self.index]     = action
        self.rewards[self.index]     = reward
        self.next_states[self.index] = next_state
        self.dones[self.index]       = done

        self.index  = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )

    def __len__(self):
        return self.size