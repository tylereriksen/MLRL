import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import gym_anytrading
import pandas as pd
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim


class DQN(nn.Module):
    
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        
    def forward(self, state):
        return self.policy_net(state)


class ReplayMemory(object):
    
    def __init__(self, capacity=100000):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQNAgent():
    
    def __init__(self, input_dim, n_actions, eps=0.9):
        self.memory = ReplayMemory()
        
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.eps = eps
        
        self.policy_net = DQN(self.input_dim, self.n_actions)
        
        self.target_net = DQN(self.input_dim, self.n_actions)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    
    def select_action(self, state, eps_decay=0.99, eps_end=0.05):
        
        sample = random.random()
        
        if sample > self.eps:
            self.eps *= eps_decay 
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        
        else:
            if self.eps > eps_end:
                self.eps *= eps_decay
                
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)
    
    
    def optimize(self, batch_size=128, gamma=0.99):
        if len(self.memory) < batch_size:
            return
        
        # get transitions from replaymemory
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        

        # boolean mapping of TF for terminal or nonterminal states
        non_terminal_states = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        # get non terminal next states
        non_terminal_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # get next state values and set terminal states to 0
        next_state_batch = torch.zeros(batch_size)
        with torch.no_grad():
            next_state_batch[non_terminal_states] = self.target_net(
                non_terminal_next_states).max(1)[0]
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        
       
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        
        # Compute the expected Q values
        expected_state_action_values = (next_state_batch * gamma) + reward_batch
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        agent.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        agent.optimizer.step()



def train(env, agent, transitions, episode, T, lr=0.01, gamma=0.99):
    
    rewards = []
    episode_reward = 0

    state, _ = env.reset()
    state = state.flatten()       
    state = torch.tensor([state], dtype=torch.float32)
    
    for t in range(T):
        
        action = agent.select_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        rewards.append(reward)
        episode_reward += reward

        if done:
            next_state = None
        else:
            next_state = next_state.flatten()
            next_state = torch.tensor([next_state], dtype=torch.float32)
            
        agent.memory.push(state, action, next_state, reward)
        agent.optimize()
        state = next_state
        
        
        
        if t % 5 == 0:
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            agent.target_net.load_state_dict(target_net_state_dict)
            
        if done: 
            break
        
    return episode_reward, info



df = pd.read_csv('gmedata.csv')
env = gym.make('stocks-v0', df=df, frame_bound=(20,500), window_size=5)
input_dim = env.observation_space.shape[1] * env.observation_space.shape[0]
n_actions = env.action_space.n
T = env.unwrapped.frame_bound[1] - env.frame_bound[0]

agent = DQNAgent(input_dim, n_actions)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

num_episodes = 300
for episode in range(num_episodes):
    reward, info = train(env, agent, Transition, episode, T)
    print(f"Episode: {episode+1}, Reward: {reward}")
    print("info: ", info)
    print()