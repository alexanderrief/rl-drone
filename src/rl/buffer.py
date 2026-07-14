import random
import numpy as np
import torch as T

class Buffer:
    def __init__(self, buffer_size, batch_size):
        """
        Initialize the rollout buffer.
        Args:
        - buffer_size (int) : total number of transitions to collect before updating.
        - batch_size  (int) : number of transitions per mini-batch (must divide buffer_size).
        Raises:
        - ValueError : if buffer_size is not divisible by batch_size.
        """
        if buffer_size % batch_size != 0:
            raise ValueError(f"buffer_size ({buffer_size}) must be divisible by batch_size ({batch_size})")

        self.buffer_size = buffer_size
        self.batch_size  = batch_size

        self.observations = []
        self.actions      = []
        self.log_probs    = []
        self.values       = []
        self.rewards      = []
        self.dones        = []

    def store(self, obs, action, log_prob, value, reward, done):
        """
        Store a single transition from the environment.
        Args:
        - obs      : observation vector.
        - action   : action taken.
        - log_prob : log probability of the action under the current policy.
        - value    : estimated state value V(s).
        - reward   : scalar reward received.
        - done     : whether the episode terminated after this step.
        """
        self.observations.append(obs)
        self.actions     .append(action)
        self.log_probs   .append(log_prob)
        self.values      .append(value)
        self.rewards     .append(reward)
        self.dones       .append(done)

    def generate_batches(self):
        """
        Convert stored transitions into shuffled mini-batches for PPO update.
        Returns:
        - observations (T.Tensor, shape (buffer_size, obs_dim))
        - actions      (T.Tensor, shape (buffer_size, action_dim))
        - log_probs    (T.Tensor, shape (buffer_size,))
        - values       (T.Tensor, shape (buffer_size,))
        - rewards      (T.Tensor, shape (buffer_size,))
        - dones        (T.Tensor, shape (buffer_size,))
        - batches      (list of index lists, each of length batch_size)
        """
        observations = T.tensor(np.array(self.observations), dtype=T.float32)
        actions      = T.tensor(np.array(self.actions),      dtype=T.float32)
        log_probs    = T.tensor(self.log_probs,              dtype=T.float32)
        values       = T.tensor(self.values,                 dtype=T.float32)
        rewards      = T.tensor(self.rewards,                dtype=T.float32)
        dones        = T.tensor(self.dones,                  dtype=T.float32)

        indices = list(range(self.buffer_size))
        random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in range(0, self.buffer_size, self.batch_size)]

        return observations, actions, log_probs, values, rewards, dones, batches

    def clear(self):
        self.observations.clear()
        self.actions     .clear()
        self.log_probs   .clear()
        self.values      .clear()
        self.rewards     .clear()
        self.dones       .clear()