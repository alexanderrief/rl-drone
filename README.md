# rl-drone

A drone trained with Proximal Policy Optimization (PPO) to fly toward a randomly placed marker in a PyBullet simulation.

The agent receives the drone's relative position and velocity as input and outputs position offsets, which a PID controller converts to motor RPMs.

## How it works

**Environment** (`FollowMarkerEnv`)
- The marker spawns at a random position each episode
- Observation: relative position to marker (x, y, z) + drone velocity (vx, vy, vz)
- Action: position offset [dx, dy, dz], clipped to ±1 m
- Reward: negative Euclidean distance to the marker, with a small penalty for flying away from it

**PPO Agent**
- Actor: two-layer MLP outputting a Gaussian distribution over actions
- Critic: two-layer MLP estimating state value V(s)
- Advantage estimation via GAE (λ = 0.95)
- Clipped surrogate objective (ε = 0.1) with entropy bonus

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py -r my_run
```

Resume from a checkpoint:

```bash
python train.py -r my_run -l my_run
```

Checkpoints are saved to `parameters/<run_name>/`.

## Evaluation

A pretrained checkpoint is included in `pretrained/`. To run it directly:

```bash
python show_off.py -l pretrained
```

Or use your own checkpoint:

```bash
python show_off.py -l my_run
```

Opens the PyBullet GUI and plots drone position and speed over 2000 steps.
