import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.rl.agent import Agent
from src.sim.env.follow_marker import FollowMarkerEnv

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", type=str, default=None, help="Checkpoint to load")
args = parser.parse_args()

env   = FollowMarkerEnv(gui=True)
agent = Agent(env, obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])

path = agent.load(args.load)
if path is None:
    print("No checkpoint found. Run train.py first.")
    env.close()
    exit()

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

C_X = "#2a78d6"
C_Y = "#eb6834"
C_Z = "#1baf7a"

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.patch.set_facecolor("#fcfcfb")

for ax in axes:
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.yaxis.label.set_color("#52514e")
    ax.xaxis.label.set_color("#52514e")

axes[0].plot(t, log_pos[:, 0], color=C_X, linewidth=1.5, label="x")
axes[0].plot(t, log_pos[:, 1], color=C_Y, linewidth=1.5, label="y")
axes[0].plot(t, log_pos[:, 2], color=C_Z, linewidth=1.5, label="z")
axes[0].axhline(env.marker_pos[0], color=C_X, linestyle="--", linewidth=1.2, alpha=0.5, label="target x")
axes[0].axhline(env.marker_pos[1], color=C_Y, linestyle="--", linewidth=1.2, alpha=0.5, label="target y")
axes[0].axhline(env.marker_pos[2], color=C_Z, linestyle="--", linewidth=1.2, alpha=0.5, label="target z")
axes[0].set_ylabel("Position [m]")
axes[0].set_title("Drone state over time", color="#0b0b0b", fontweight="medium")
axes[0].legend(loc="right", framealpha=0.9, edgecolor="#e1e0d9", labelcolor="#52514e")

axes[1].plot(t, np.linalg.norm(log_vel, axis=1), color=C_X, linewidth=1.5)
axes[1].set_ylabel("Speed [m/s]")
axes[1].set_xlabel("Step")

plt.tight_layout()
plt.savefig(f"{path}/flight_data.png", facecolor="#fcfcfb")
plt.show()
