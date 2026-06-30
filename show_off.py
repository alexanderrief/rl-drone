import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.rl.agent import Agent
from src.sim.env.follow_marker import FollowMarkerEnv
from src.sim.env.visual_follow_marker import VisualFollowMarkerEnv

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", type=str, default=None, help="Parameters to load")
parser.add_argument("-v", "--visual", action="store_true", help="Use VisualFollowMarkerEnv")
args = parser.parse_args()

EnvClass = VisualFollowMarkerEnv if args.visual else FollowMarkerEnv
env = EnvClass(gui=True)

obs_dim    = 3 if args.visual else 6
action_dim = 4 if args.visual else 3

agent = Agent(env, obs_dim=obs_dim, action_dim=action_dim)

path = agent.load(args.load)
agent.policy.eval()

obs, _ = env.reset()

log_pos  = []
log_vel  = []
log_dist = []

for _ in range(2000):
    action = agent.act(obs)
    obs, _, terminated, truncated, _ = env.step(action)

    state = env._getDroneStateVector(0)
    log_pos.append(state[0:3].copy())
    log_vel.append(state[10:13].copy())
    log_dist.append(np.linalg.norm(state[0:3] - env.marker_pos))

    if terminated or truncated:
        obs, _ = env.reset()

env.close()

log_pos  = np.array(log_pos)
log_vel  = np.array(log_vel)
log_dist = np.array(log_dist)
t        = np.arange(len(log_dist))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t, log_pos[:, 0], label="x")
axes[0].plot(t, log_pos[:, 1], label="y")
axes[0].plot(t, log_pos[:, 2], label="z")
axes[0].axhline(env.marker_pos[0], color='r', linestyle='--', alpha=0.3, label="marker x")
axes[0].axhline(env.marker_pos[1], color='g', linestyle='--', alpha=0.3, label="marker y")
axes[0].axhline(env.marker_pos[2], color='b', linestyle='--', alpha=0.3, label="marker z")
axes[0].set_ylabel("Position [m]")
axes[0].legend(loc="right")
axes[0].set_title("Drone state over time")

axes[1].plot(t, np.linalg.norm(log_vel, axis=1))
axes[1].set_ylabel("Speed [m/s]")
axes[1].set_xlabel("Step")

plt.tight_layout()
plt.savefig(f"{path}/flight_data.png")
plt.show()
