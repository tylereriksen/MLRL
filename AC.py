# import necessary packages
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from collections import deque


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)


def actor_critic(env, actor, critic, episode, actor_optimizer, critic_optimizer, gamma=0.99, T=1000):

    state, _ = env.reset()

    log_prob_actions = []
    values = []
    rewards = []
    episode_reward = 0

    for t in range(T):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = actor(state)
        value = critic(state)
        dist = Categorical(probs)
        action = dist.sample()
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward

        log_prob_action = dist.log_prob(action)
        log_prob_actions.append(log_prob_action)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float))

        if done:
            print(f"Episode {episode} finished with total reward {episode_reward}")
            break
        state = next_state

    actor_loss, critic_loss = compute_loss(log_prob_actions, values, rewards, gamma)

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    return episode_reward


def compute_loss(log_prob_actions, values, rewards, gamma):
    returns = []
    Gt = 0
    pw = 0
    for reward in reversed(rewards):
        Gt = reward + gamma ** pw * Gt
        pw += 1
        returns.insert(0, Gt)

    log_prob_actions = torch.cat(log_prob_actions)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values
    actor_loss = -(log_prob_actions * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    return actor_loss, critic_loss


env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=0.01)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.01)


NUM_EPISODES = 10_000
MAX_TIMESTEPS = 1_000
AVERAGE_REWARD_TO_SOLVE = 200
NUM_EPS_TO_SOLVE = 100
GAMMA = 0.995

scores_last_timesteps = deque([], NUM_EPS_TO_SOLVE)


# benchmark rewards
train_rewards = []
for i_episode in range(100):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    total_rewards = 0
    for t in range(MAX_TIMESTEPS):
        probs = actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_rewards += reward
        if done:
            print('Episode {} ended with total reward of {}'.format(i_episode, total_rewards))
            train_rewards.append(total_rewards)
            break

print('Average reward before training was {}'.format(np.mean(np.array(train_rewards))))

AVERAGE_REWARD_TO_SOLVE = max(AVERAGE_REWARD_TO_SOLVE, 2.5 * np.mean(np.array(train_rewards)))

test_rewards = []

for episode in range(NUM_EPISODES):
    if episode >= NUM_EPS_TO_SOLVE:
        if (sum(scores_last_timesteps) / NUM_EPS_TO_SOLVE > AVERAGE_REWARD_TO_SOLVE):
            print("solved after {} episodes".format(episode))
            break
    episode_reward = actor_critic(env, actor, critic, episode, actor_optimizer, critic_optimizer, GAMMA, MAX_TIMESTEPS)
    test_rewards.append(episode_reward)
    scores_last_timesteps.append(episode_reward)

plt.plot(range(len(test_rewards)), test_rewards, 'b')
plt.plot(range(len(test_rewards)), [np.mean(np.array(train_rewards)) for _ in range(len(test_rewards))],
         linestyle='--', color="red", alpha=0.2, label='Pre- Avg Rewards')
plt.plot(range(len(test_rewards)), [AVERAGE_REWARD_TO_SOLVE for _ in range(len(test_rewards))],
         linestyle='--', color="gold", alpha=0.5, label='Target Post- Avg Rewards')
plt.legend(loc="upper left")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards over Episodes during Training")
plt.show()


env.close()
