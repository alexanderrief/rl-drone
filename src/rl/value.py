import torch as T
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: T.Tensor) -> T.Tensor:
        """
        Returns the estimated state value V(s) for a given observation.
        Args:
        - obs (T.Tensor, shape (batch, input_dim)) : observation vector.
        Returns:
        - T.Tensor, shape (batch, 1) : estimated value for each observation.
        """
        return self.network(obs)
