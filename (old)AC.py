# import necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import matplotlib.pyplot as plt
import gym
import numpy as np

'''
Seems to have a lot of variance in training
Random Dips in rewards in training seem to be a common thing for Actor-Critic
'''

# define some constants
ENV_NAME = 'CartPole-v1'
LAYER1 = 128
LAYER2 = 128
DISCOUNT = 0.99
LEARNING_RATE = 0.001
EPISODES = 500

# Neural Net
class NeuralNet(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(NeuralNet, self).__init__()

        # define the layers
        self.fc1 = nn.Linear(state_size, LAYER1)
        self.fc2 = nn.Linear(LAYER1, LAYER2)
        self.fc3 = nn.Linear(LAYER2, action_size)

    # step through function
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4)
        x = self.fc3(x)
        return x

# define the Actor-Critic Model
class ActorandCritic(nn.Module):
    def __init__(self, actor: NeuralNet, critic: NeuralNet):
        super().__init__()
        self.actor = actor
        self.critic = critic

    #  output prediction function
    def forward(self, state):
        policy_output = self.actor(state)
        value_output = self.critic(state)
        return policy_output, value_output


# get all the rewards collected and reverse propagate with the Values
def compute_returns(rewards: list, discount=DISCOUNT):
    computed_returns = []
    accumulated = 0
    for idx, val in enumerate(reversed(rewards)):
        computed_returns.insert(0, val + discount * accumulated)
        accumulated = computed_returns[0]
    computed_returns = torch.tensor(computed_returns)
    return computed_returns


# update to get the losses
def update(returns, log_prob_actions, values, optimizer):
    returns = returns.detach()
    delta = returns - values
    # policy_loss = - (returns * log_prob_actions).sum()
    policy_loss = - (delta * log_prob_actions).sum() # calculate the policy loss
    # value_loss = F.smooth_l1_loss(returns, values).sum() # calculate the value_loss
    value_loss = (delta ** 2).sum()
    optimizer.zero_grad()

    total_loss = policy_loss + value_loss
    total_loss.backward() # get the loss gradient

    optimizer.step()
    return policy_loss.item(), value_loss.item()


# Create the environment
env = gym.make(ENV_NAME)

# Observation Space: cart position, cart velocity, pole angle, pole angular velocity
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(env.observation_space.high) # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low) # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

# define the variables
actor, critic = NeuralNet(state_size, action_size), NeuralNet(state_size, 1) # actor gives action probs
agentPol = ActorandCritic(actor, critic)
agentPol.load_state_dict(agentPol.state_dict())
optimizer = optim.Adam(agentPol.parameters(), lr=LEARNING_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# iterate through the different episodes
train_rewards = []
for i in range(EPISODES):
    agentPol.train()
    log_prob_actions, values, rewards = [], [], []
    done = False
    total_reward = 0
    state = env.reset()

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        policy_output, value_output = agentPol(state) # get the predictions
        action_prob = F.softmax(policy_output, dim=-1) # get the action probabilities
        distribution = distributions.Categorical(action_prob)
        action = distribution.sample() # get a sample from the action probabilities distribution

        log_prob_action = distribution.log_prob(action) # get the log of the probability of the action chosen
        state, reward, done, _ = env.step(action.item()) # take the action as a step in environment

        # add these values to their respective lists
        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        values.append(value_output)

        total_reward += reward # add rewards

    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    computed_returns = compute_returns(rewards, DISCOUNT)
    loss = update(computed_returns, log_prob_actions, values, optimizer)

    train_rewards.append(total_reward)
    print('episode: ', i)
    print('train reward: ', total_reward)

plt.plot(range(len(train_rewards)), train_rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards Collected Over Episodes")

average_reward = []
for idx in range(len(train_rewards)):
    avg_list = np.empty(shape=(1, ), dtype=int)
    if idx < 50:
        avg_list = train_rewards[: idx + 1]
    else:
        avg_list = train_rewards[idx - 49: idx + 1]
    average_reward.append(np.average(avg_list))

plt.plot(average_reward, 'r-') # moving average
plt.show()

