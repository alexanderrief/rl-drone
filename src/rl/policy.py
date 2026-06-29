import torch as T
import torch.nn as nn
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    def __init__ (self, input_dim: int = 6, output_dim: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.log_std = nn.Parameter(T.zeros(output_dim))

    def forward(self, obs):
        mean = self.network(obs)
        log_std = self.log_std
        return Normal(mean, T.exp(log_std))

        