import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

env = gym.make('CartPole-v1')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def apply_softmax(self, x):
        return F.softmax(x, dim=1)

    def choose_action(self, probs):
        action = torch.multinomial(probs, 1).item()
        return action

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



def calculate_cumulative_rewards(rewards):
    returns = []
    tot = 0
    for r in reversed(rewards):
        tot = r + 0.99 * tot
        returns.insert(0, tot)
    return returns


DISCOUNT = 0.99
LEARNING_RATE = 0.01


# Usage example
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 64

policy_net = NeuralNet(input_size, hidden_size, output_size)


for episode in range(100):
    done = False
    state = env.reset()
    state = torch.from_numpy(np.array(state)).float()
    state = state.unsqueeze(0)
    episode_rewards = []
    episode_actions = []
    episode_states = [state]
    total_rewards = 0
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
    #print(f'States were {str(episode_states)}')
    #print(f'Rewards were {str(episode_rewards)}')
    #print(f'Actions were {str(episode_actions)}')
    #print()


env.close()





