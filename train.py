import argparse
import matplotlib.pyplot as plt
from src.rl.agent import Agent
from src.sim.env.follow_marker import FollowMarkerEnv

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run",  type=str, default=None, help="Name for this training run")
parser.add_argument("-l", "--load", type=str, default=None, help="Checkpoint to resume from")
args = parser.parse_args()

epochs       = 500
batch_epochs = 4

env   = FollowMarkerEnv()
agent = Agent(env, obs_dim=6, action_dim=3)

if args.load:
    agent.load(args.load)

reward_history = []

for epoch in range(epochs):
    agent.collect()
    agent.update(batch_epochs)

    avg_reward = sum(agent.buffer.rewards) / len(agent.buffer.rewards)
    reward_history.append(avg_reward)

    if not epoch % 10:
        agent.info(epoch, epochs)

path = agent.save(args.run)

plt.plot(reward_history)
plt.xlabel("Epoch")
plt.ylabel("Avg Reward")
plt.title("PPO Training")
plt.savefig(f"{path}/reward_data.png")
plt.show()
