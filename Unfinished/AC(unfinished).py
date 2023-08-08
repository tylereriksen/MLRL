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
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lr = lr

    # forward passing through the Neural Net
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4)
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
        softmax_probs = F.log_softmax(output, dim=1)

        # Select the log-probability of the chosen action (action is a scalar here)
        log_prob_selected_action = softmax_probs[0, action]

        # Compute gradients of the selected log-probability with respect to the neural network parameters
        log_prob_selected_action.backward()

        # Access gradients
        gradients = []
        for param in self.parameters():
            gradients.append(param.grad.clone())

        # Clear gradients for next computation
        self.zero_grad()

        return gradients



# custom Neural Net Class
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lr = lr

    # forward passing through the Neural Net
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.dropout(x, p=0.4)
        x = self.fc3(x)
        return x

    def get_val_gradient(self, state):
        # Pass state through the neural network
        output = self.forward(state)

        output.backward()

        # Access gradients
        gradients = []
        for param in self.parameters():
            gradients.append(param.grad.clone())

        # Clear gradients for next computation
        self.zero_grad()

        return gradients








HIDDEN_SIZE = 128
DISCOUNT = 0.99
EPISODES = 1000


input_size = env.observation_space.shape[0]
output_size = env.action_space.n

actor_net = Actor(input_size, HIDDEN_SIZE, output_size, 0.001)
critic_net = Critic(input_size + 1, HIDDEN_SIZE, 1, 0.001)


for episode in range(EPISODES):
    done = False
    state = env.reset()
    state = torch.from_numpy(np.array(state)).float()
    state = state.unsqueeze(0)
    total_rewards = 0
    action = actor_net.choose_action(actor_net.apply_softmax(actor_net.forward(state)))
    value_to_add = torch.tensor([[action]])
    state_action = torch.cat((state, value_to_add), dim=1)

    while not done:
        new_state, reward, done, _ = env.step(action)
        new_state = torch.from_numpy(np.array(new_state)).float()
        new_state = new_state.unsqueeze(0)
        new_action = actor_net.choose_action(actor_net.apply_softmax(actor_net.forward(new_state)))
        new_value_to_add = torch.tensor([[new_action]])
        new_state_action = torch.cat((new_state, new_value_to_add), dim=1)

        #updating parameters
        gradient_actor = actor_net.get_log_prob_gradient(state, action)
        critic_output = critic_net.forward(state_action)[0]
        with torch.no_grad():
            for param, gradient in zip(actor_net.parameters(), gradient_actor):
                param += actor_net.lr * critic_output * gradient

        G_t = reward + DISCOUNT * critic_net.forward(new_state_action)[0] - critic_output
        gradient_critic = critic_net.get_val_gradient(state_action)
        with torch.no_grad():
            for param, gradient in zip(critic_net.parameters(), gradient_critic):
                param += critic_net.lr * G_t * gradient

        state = new_state
        action = new_action
        state_action = new_state_action
        total_rewards += reward

    print(f'Episode {episode}: {total_rewards}')







