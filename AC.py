# Import necessary packages
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

'''
Observation: there seems to be a lot more variance in the total_rewards
'''

# Define constants
ENV_NAME = 'CartPole-v1'
LAYER1 = 64
LAYER2 = 64
EPISODES = 1000
DISCOUNT = 0.999
LEARNING_RATE = 0.001

# Define the actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, LAYER1)
        self.fc2 = nn.Linear(LAYER1, LAYER2)
        self.fc_pi = nn.Linear(LAYER2, action_dim)
        self.fc_v = nn.Linear(LAYER2, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)
        v = self.fc_v(x)

        return pi, v


# Define the environment and other parameters
env = gym.make(ENV_NAME)

# Initialize the ActorCritic network
agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)

# Define the optimizer
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

total_rewards = []

# Define the training loop
for episode in range(EPISODES):
    # Initialize the environment
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select an action using the agent's policy
        probs, val = agent(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(np.arange(len(probs)), p=probs.detach().numpy())

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Calculate the TD error and loss
        _, next_val = agent(torch.tensor(next_state, dtype=torch.float32))
        expected_return = reward + DISCOUNT * next_val * (1 - done)
        td_error = expected_return - val
        actor_loss = -torch.log(probs[action]) * td_error
        critic_loss = torch.square(td_error)
        loss = actor_loss + critic_loss

        # Update the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Set the state to the next state
        state = next_state
        if total_reward >= 500: # capping it at 500 since this environment can go as high as it can
            break

    # Print the total reward for the episode
    print(f'Episode {episode}: Total reward = {total_reward}')
    total_rewards.append(total_reward)

plt.plot(range(len(total_rewards)), total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards over Episodes')
plt.show()
