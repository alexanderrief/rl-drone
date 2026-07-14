import os
import torch as T
import numpy as np
from datetime import datetime

from src.rl.buffer import Buffer
from src.rl.policy import PolicyNetwork
from src.rl.value import ValueNetwork
from src.sim.env.follow_marker import FollowMarkerEnv

class Agent:
    def __init__(self,
                 env:          FollowMarkerEnv,
                 obs_dim:      int,
                 action_dim:   int,
                 epsilon:      float = 0.1,
                 gamma:        float = 0.95,
                 gae_lambda:   float = 0.95,
                 buffer_size:  int   = 2048,
                 batch_size:   int   = 64,
                 lr:           float = 3e-4, 
                 entropy_coef: float = 1e-3,
                 device:       str   = "cuda"):
        """
        Initialize the PPO agent.
        Args:
        - env          (FollowMarkerEnv) : the simulation environment.
        - obs_dim      (int)   : dimension of the observation vector.
        - action_dim   (int)   : dimension of the action vector.
        - epsilon      (float) : PPO clip range for the policy update.
        - gamma        (float) : discount factor for future rewards.
        - gae_lambda   (float) : smoothing factor for GAE advantage estimation.
        - buffer_size  (int)   : number of transitions to collect before each update.
        - batch_size   (int)   : number of transitions per mini-batch.
        - lr           (float) : learning rate for both policy and value optimizer.
        - entropy_coef (float) : weight of the entropy bonus in the policy loss.
        - device       (str)   : target device; falls back to "cpu" if CUDA is unavailable.
        """
        self.env          = env
        self.epsilon      = epsilon
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.buffer_size  = buffer_size
        self.batch_size   = batch_size
        self.entropy_coef = entropy_coef
        self.device = device if T.cuda.is_available() else "cpu"
        
        self.policy = PolicyNetwork(input_dim=obs_dim, output_dim=action_dim).to(self.device)
        self.value  = ValueNetwork(input_dim=obs_dim).to(self.device)
        self.buffer = Buffer(buffer_size, batch_size)

        self.policy_opt = T.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_opt  = T.optim.Adam(self.value.parameters(), lr=lr)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Select a deterministic action for a given observation.
        Args:
        - obs (np.ndarray, shape (obs_dim,)) : observation vector.
        Returns:
        - np.ndarray, shape (action_dim,) : mean action from the policy distribution.
        """
        with T.no_grad():
            obs_t = T.tensor(obs, dtype=T.float32).to(self.device)
            return self.policy(obs_t).mean.cpu().numpy()

    def collect(self):
        """
        Run the environment for buffer_size steps and store transitions in the buffer.
        Notes:
        - Appends one extra value estimate at the end (last_value) for GAE computation.
        - Resets the environment on episode termination and continues collecting.
        """
        self.buffer.clear()
        obs, _ = self.env.reset()
        
        with T.no_grad():
            for _ in range(self.buffer_size):
                obs_t    = T.tensor(obs, dtype=T.float32).to(self.device)
                dist     = self.policy(obs_t)
                action   = dist.rsample()
                log_prob = dist.log_prob(action).sum(-1)
                value    = self.value(obs_t)

                action = action.cpu().numpy()
                obs_next, reward, done, truncated, _ = self.env.step(action)

                self.buffer.store(
                    obs,
                    action,
                    log_prob.item(),
                    value.item(),
                    reward,
                    done
                )

                if done or truncated:
                    obs, _ = self.env.reset()
                else:
                    obs = obs_next

        last_value = 0 if done else self.value(T.tensor(obs_next, dtype=T.float32)
                                               .to(self.device)).item()
        self.buffer.values.append(last_value)

    def compute_advantages(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE) for each timestep.
        Args:
        - rewards (list) : collected rewards, length buffer_size.
        - values  (list) : estimated state values V(s), length buffer_size + 1 (last_value appended).
        - dones   (list) : episode termination flags, length buffer_size.
        Returns:
        - advantages (T.Tensor, shape (buffer_size,)) : advantage estimate A(s, a) for each timestep.
        """
        advantages     = T.zeros(self.buffer_size)
        rewards        = T.tensor(rewards, dtype=T.float32)
        values         = T.tensor(values,  dtype=T.float32)
        dones          = T.tensor(dones,   dtype=T.float32)
        last_advantage = 0

        for t in reversed(range(self.buffer_size)):
            delta          = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages[t]  = last_advantage

        return advantages
    
    def update(self, batch_epochs: int):
        """
        Run the PPO update over the collected rollout.
        Args:
        - batch_epochs (int) : number of passes over the buffer per update.
        """
        advantages = self.compute_advantages(self.buffer.rewards, self.buffer.values, self.buffer.dones)
        standard_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        standard_advantages = standard_advantages.to(self.device)

        for _ in range(batch_epochs):
            observations, actions, log_probs, values, _, _, batches = self.buffer.generate_batches()
            observations = observations.to(self.device)
            actions      = actions     .to(self.device)
            log_probs    = log_probs   .to(self.device)
            values       = values      .to(self.device)

            for batch in batches:
                pi           = self.policy(observations[batch])
                log_prob_new = pi.log_prob(actions[batch]).sum(-1)
                log_prob_old = log_probs[batch]
                ratio        = T.exp(log_prob_new - log_prob_old)
                standard_adv = standard_advantages[batch]
                adv          = advantages[batch]

                actor_loss  = -T.min(ratio * standard_adv, T.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * standard_adv).mean()
                values_new  = self.value(observations[batch]).squeeze()
                returns     = (adv + values[batch]).detach()
                critic_loss = (returns - values_new).pow(2).mean()
                entropy     = pi.entropy().sum(-1).mean()

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.policy_opt.step()
                self.value_opt.step()

    def save(self, run_name: str | None = None) -> str:
        """
        Save policy and value network weights to disk.
        Args:
        - run_name (str, optional) : folder name under parameters/. Defaults to current timestamp.
        """
        name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M")
        path = f"parameters/{name}"
        os.makedirs(path, exist_ok=True)
        T.save(self.policy.state_dict(), f"{path}/policy.pt")
        T.save(self.value.state_dict(), f"{path}/value.pt")
        print(f"Saved to {path}")
        return path

    def load(self, run_name: str | None = None) -> str | None:
        """
        Load policy and value network weights from disk.
        Args:
        - run_name (str, optional) : folder name under parameters/. Defaults to the most recent run.
        """
        if run_name:
            path = run_name if os.path.exists(run_name) else f"parameters/{run_name}"
        else:
            if not os.path.exists("parameters") or not os.listdir("parameters"):
                print("No parameters found, starting fresh")
                return 
            runs = sorted(os.listdir("parameters"), key=lambda r: os.path.getmtime(f"parameters/{r}"))
            path = f"parameters/{runs[-1]}"
        print(f"Loading from {path}")
        self.policy.load_state_dict(T.load(f"{path}/policy.pt", weights_only=True))
        self.value.load_state_dict(T.load(f"{path}/value.pt",  weights_only=True))
        return path

    def info(self, epoch: int, total_epochs: int) -> None:
        crashes          = int(sum(self.buffer.dones))
        mean_reward      = float(np.mean(self.buffer.rewards)) if self.buffer.rewards else 0.0
        mean_value       = float(np.mean(self.buffer.values))  if self.buffer.values  else 0.0
        mean_episode_len = self.buffer_size / crashes if crashes > 0 else self.buffer_size

        obs_t   = T.tensor(np.array(self.buffer.observations), dtype=T.float32).to(self.device)
        with T.no_grad():
            entropy = self.policy(obs_t).entropy().sum(-1).mean().item()

        print(
    f"  ┌─────────────────────────────┐\n"
    f"  │   Rollout  {epoch:>4d} / {total_epochs:<4d}      │\n"
    f"  ├─────────────────────────────┤\n"
    f"  │  Mean Reward    {mean_reward:+8.3f}    │\n"
    f"  │  Mean Value     {mean_value:+8.3f}    │\n"
    f"  │  Entropy        {entropy:8.3f}    │\n"
    f"  │  Crashes        {crashes:8d}    │\n"
    f"  └─────────────────────────────┘"
)

