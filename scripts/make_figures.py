"""
Evaluate the pretrained policy headless (PyBullet DIRECT) and write the README figures:
  - docs/evaluation.png : distance to target over time + top-down trajectories
  - docs/episode.gif    : one rendered episode (software renderer, no display needed)

Run from the repository root:
    python scripts/make_figures.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rl.agent import Agent
from src.sim.env.follow_marker import FollowMarkerEnv

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load",     type=str,   default="pretrained", help="Checkpoint to evaluate")
parser.add_argument("-n", "--episodes", type=int,   default=20,           help="Number of evaluation episodes")
parser.add_argument("-t", "--seconds",  type=float, default=30.0,         help="Episode length [s]")
parser.add_argument("-s", "--seed",     type=int,   default=0,            help="Seed for the target positions")
parser.add_argument("-o", "--out",      type=str,   default="docs",       help="Output folder")
parser.add_argument("--no-gif", action="store_true", help="Skip rendering the GIF")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

env       = FollowMarkerEnv(gui=False)
env.rng   = np.random.default_rng(args.seed)
agent     = Agent(env, obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
if agent.load(args.load) is None:
    sys.exit("No checkpoint found.")
agent.policy.eval()

steps = int(args.seconds * env.CTRL_FREQ)
dt    = 1.0 / env.CTRL_FREQ


def add_sphere(env: FollowMarkerEnv, pos, radius: float, rgba: list) -> None:
    """Draw a sphere (visual only, no collision shape, so it has no effect on the dynamics)."""
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=env.CLIENT)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=list(pos),
                      physicsClientId=env.CLIENT)


def run_episode(render_every: int = 0):
    obs, _ = env.reset()
    frames, positions, distances = [], [], []
    view = proj = None
    if render_every:
        add_sphere(env, env.marker_pos, 0.04, [0.84, 0.27, 0.27, 1.0])          # target
        start  = env._getDroneStateVector(0)[0:3]
        center = (start + env.marker_pos) / 2
        span   = np.linalg.norm(env.marker_pos - start)
        view = p.computeViewMatrixFromYawPitchRoll(center.tolist(), distance=0.5 + 0.9 * span,
                                                   yaw=30, pitch=-25, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=4 / 3, nearVal=0.05, farVal=20)

    for k in range(steps):
        obs, _, terminated, _, _ = env.step(agent.act(obs))
        state = env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        distances.append(np.linalg.norm(state[0:3] - env.marker_pos))
        if render_every and k % render_every == 0:
            add_sphere(env, state[0:3], 0.008, [0.16, 0.47, 0.84, 1.0])          # trail
            _, _, rgb, _, _ = p.getCameraImage(480, 360, viewMatrix=view, projectionMatrix=proj,
                                               renderer=p.ER_TINY_RENDERER, physicsClientId=env.CLIENT)
            frames.append(Image.fromarray(np.reshape(rgb, (360, 480, 4))[:, :, :3].astype(np.uint8)))
        if terminated:
            break
    return np.array(positions), np.array(distances), env.marker_pos.copy(), terminated, frames


# --- Evaluation episodes -----------------------------------------------------
results = [run_episode() for _ in range(args.episodes)]

final  = np.array([d[-1] for _, d, _, _, _ in results])
starts = np.array([np.linalg.norm(pos[0] - m) for pos, _, m, _, _ in results])
crash  = sum(term for _, _, _, term, _ in results)
reach  = [np.argmax(d < 0.1) * dt for _, d, _, _, _ in results if (d < 0.1).any()]
print(f"episodes: {args.episodes}, length: {args.seconds:.0f} s, terminated early: {crash}")
print(f"initial distance: {starts.min():.2f} - {starts.max():.2f} m")
print(f"final distance:   median {np.median(final) * 100:.1f} cm, max {final.max() * 100:.1f} cm")
print(f"within 10 cm:     {len(reach)}/{args.episodes} episodes, after {min(reach):.1f} - {max(reach):.1f} s")

# --- Figure ------------------------------------------------------------------
BG, GRID, AXIS, TEXT = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#52514e"
C_LINE, C_TARGET     = "#2a78d6", "#d64545"

fig, (ax_d, ax_xy) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
fig.patch.set_facecolor(BG)
for ax in (ax_d, ax_xy):
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

for _, d, _, _, _ in results:
    ax_d.plot(np.arange(len(d)) * dt, d, color=C_LINE, linewidth=1.0, alpha=0.45)
ax_d.axhline(0.1, color=TEXT, linestyle="--", linewidth=1.0, alpha=0.6)
ax_d.text(args.seconds, 0.13, "10 cm", color=TEXT, ha="right", va="bottom", fontsize=9)
ax_d.set_xlabel("Time [s]")
ax_d.set_ylabel("Distance to target [m]")
ax_d.set_title(f"Distance to target, {args.episodes} episodes with random targets", color="#0b0b0b")

for pos, _, m, _, _ in results:
    ax_xy.plot(pos[:, 0], pos[:, 1], color=C_LINE, linewidth=1.0, alpha=0.6)
    ax_xy.scatter(m[0], m[1], color=C_TARGET, s=18, zorder=3)
ax_xy.scatter(results[0][0][0, 0], results[0][0][0, 1], color="#0b0b0b", marker="s", s=30, zorder=4, label="start")
ax_xy.scatter([], [], color=C_TARGET, s=18, label="target")
ax_xy.set_aspect("equal")
ax_xy.set_xlim(-2.2, 2.2)
ax_xy.set_ylim(-2.2, 2.2)
ax_xy.set_xlabel("x [m]")
ax_xy.set_ylabel("y [m]")
ax_xy.set_title("Trajectories (top view)", color="#0b0b0b")
ax_xy.legend(loc="lower right", framealpha=0.9, edgecolor=GRID, labelcolor=TEXT)

plt.tight_layout()
plt.savefig(f"{args.out}/evaluation.png", dpi=110, facecolor=BG)
print(f"saved {args.out}/evaluation.png")

# --- GIF -------------------------------------------------------------------
if not args.no_gif:
    render_every = 6                                         # 5 frames per simulated second
    _, d, _, _, frames = run_episode(render_every=render_every)
    reached = np.argmax(d < 0.05) if (d < 0.05).any() else len(d) - 1
    frames  = frames[: reached // render_every + 10]         # stop shortly after reaching the target
    frames  = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(f"{args.out}/episode.gif", save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)      # 10 fps = 2x real time
    print(f"saved {args.out}/episode.gif ({len(frames)} frames, start distance {d[0]:.2f} m)")

env.close()
