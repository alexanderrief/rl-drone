import torch as T
import torch.nn as nn
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    def __init__ (self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.log_std = nn.Parameter(T.zeros(output_dim))

    def forward(self, obs: T.Tensor) -> Normal:
        """
        Compute the policy distribution pi(a|s) for a given observation.
        Args:
        - obs (T.Tensor, shape (batch, 6)) : observation vector.
        Returns:
        - Normal distribution over actions, shape (batch, 3).
        """
        mean = self.network(obs)
        log_std = self.log_std
        return Normal(mean, T.exp(log_std))

        