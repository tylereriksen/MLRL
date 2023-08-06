import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical


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
            print(
                f"Episode {episode} finished with total reward {episode_reward}")
            break
        state = next_state

    actor_loss, critic_loss = compute_loss(
        log_prob_actions, values, rewards, gamma)

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

rewards = []

for episode in range(300):
    episode_reward = actor_critic(
        env, actor, critic, episode, actor_optimizer, critic_optimizer)
    rewards.append(episode_reward)

plt.figure(figsize=(12, 8))
plt.plot(rewards, label='Reward per episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards over episodes')
plt.legend()
plt.show()
