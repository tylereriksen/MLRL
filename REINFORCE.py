# import the
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt

# get the environment
env = gym.make('CartPole-v1')


# custom Neural Net Class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    # forward passing through the Neural Net
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # applying the softmax layer to get probabilities
    def apply_softmax(self, x):
        return F.softmax(x, dim=1)

    # choose action based on the probability distribution given
    def choose_action(self, probs):
        action = torch.multinomial(probs, 1).item()
        return action

    # get the gradients with respect to ∇_θ log π(A_t|S_t, θ)
    def get_log_prob_gradient(self, state, action):

        # Pass state through the neural network
        output = self.forward(state)

        # Apply softmax to obtain action probabilities
        softmax_probs = self.apply_softmax(output)

        # Calculate log-probabilities of actions
        log_probs = torch.log(softmax_probs)

        # Select the log-probability of the chosen action (action is a scalar here)
        log_prob_selected_action = log_probs[0, action]

        # Compute gradients of the selected log-probability with respect to the neural network parameters
        log_prob_selected_action.backward()

        # Access gradients
        gradients = []
        for param in self.parameters():
            gradients.append(param.grad.clone())

        # Clear gradients for next computation
        self.zero_grad()

        return gradients


# calculate the cumulated rewards G_t
def calculate_cumulative_rewards(rewards):
    returns = []
    tot = 0
    for r in reversed(rewards):
        tot = r + 0.99 * tot
        returns.insert(0, tot)
    return returns


# Define the constants for training and testing
DISCOUNT = 0.995
LEARNING_RATE = 0.001
EPISODES = 5000
NUM_EPS_TO_SOLVE = 100
AVERAGE_REWARD_TO_SOLVE = 200
HIDDEN_SIZE = 64



scores_last_timesteps = deque([], NUM_EPS_TO_SOLVE)
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
policy_net = NeuralNet(input_size, HIDDEN_SIZE, output_size)


# test the randomly initialized Neural Net to see the average reward per episode
baseline_rewards = []
for episode in range(100):
    done = False
    state = env.reset()
    state = torch.from_numpy(np.array(state)).float()
    state = state.unsqueeze(0)
    total_rewards = 0
    while not done:
        action = policy_net.choose_action(policy_net.apply_softmax(policy_net.forward(state)))
        new_state, reward, done, _ = env.step(action)
        total_rewards += reward
        new_state = torch.from_numpy(np.array(new_state)).float()
        new_state = new_state.unsqueeze(0)
        state = new_state
    baseline_rewards.append(total_rewards)

print(f'Average reward collected before training is {np.mean(np.array(baseline_rewards))}')


# train the model and see the average per episode over a certain number of last episodes
training_rewards = []
for episode in range(EPISODES):
    done = False
    state = env.reset()
    state = torch.from_numpy(np.array(state)).float()
    state = state.unsqueeze(0)
    
    # keep track of the episode's rewards, states, and actions
    episode_rewards = []
    episode_actions = []
    episode_states = [state]
    total_rewards = 0
    if episode >= NUM_EPS_TO_SOLVE:
        if (sum(scores_last_timesteps) / NUM_EPS_TO_SOLVE > AVERAGE_REWARD_TO_SOLVE):
            print("solved after {} episodes".format(episode))
            break
        
    # generate an episode in its entirety
    while not done:
        action = policy_net.choose_action(policy_net.apply_softmax(policy_net.forward(state)))
        new_state, reward, done, _ = env.step(action)
        total_rewards += reward
        new_state = torch.from_numpy(np.array(new_state)).float()
        new_state = new_state.unsqueeze(0)
        episode_rewards.append(reward)
        episode_actions.append(action)
        episode_states.append(new_state)
        state = new_state

    # update the parameters in the neural net according to the pseudocode
    discounted_values = calculate_cumulative_rewards(episode_rewards)
    for i in range(len(episode_actions)):
        curr_update_state = episode_states[i]
        curr_update_reward = discounted_values[i]
        curr_update_action = episode_actions[i]

        gradients = policy_net.get_log_prob_gradient(curr_update_state, curr_update_action)

        with torch.no_grad():
            for param, gradient in zip(policy_net.parameters(), gradients):
                param += LEARNING_RATE * (DISCOUNT ** (i + 1)) * curr_update_reward * gradient
    print(f'Total Reward for Episode {episode} was {total_rewards}')
    scores_last_timesteps.append(total_rewards)
    training_rewards.append(total_rewards)


env.close()

# graph the rewards collected for each episode during training process
plt.plot(range(len(training_rewards)), [np.mean(np.array(baseline_rewards)) for _ in range(len(training_rewards))],
         linestyle='--', color='red', alpha=0.5, label='Pre- Avg Rewards')
plt.plot(range(len(training_rewards)), [AVERAGE_REWARD_TO_SOLVE for _ in range(len(training_rewards))],
         linestyle='--', color="gold", alpha=0.5, label='Target Post- Avg Rewards')
plt.plot(range(len(training_rewards)), training_rewards, 'b')
plt.legend(loc="upper left")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards over Episodes during Training")
plt.show()





