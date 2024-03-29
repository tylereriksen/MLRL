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

HIDDEN_LAYER1 = 64
HIDDEN_LAYER2 = 64
REPLAY_BUFF_SIZE = 1000
EPSILON = 1.0
MIN_EPS = 0.01
EXPLORE_FRAC = 0.3
BATCH_SIZE = 128
GAMMA = 0.995
LEARN_RATE = 0.005
UPDATE_EVERY = 100
EPISODES = 250


class DQN(nn.Module):
    
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_LAYER1),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER1, HIDDEN_LAYER2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER2, n_actions)
        )
        
    def forward(self, state):
        return self.policy_net(state)


class ReplayMemory(object):
    
    def __init__(self, capacity=REPLAY_BUFF_SIZE):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQNAgent():
    
    def __init__(self, input_dim, n_actions, eps=EPSILON, lr=LEARN_RATE):
        self.memory = ReplayMemory()
        
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.eps = eps
        
        self.policy_net = DQN(self.input_dim, self.n_actions)
        self.target_net = DQN(self.input_dim, self.n_actions)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.t = 1
        
    
    def select_action(self, state):
        
        sample = random.random()
        
        if sample > self.eps:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
                
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)
    
    
    def optimize(self, batch_size=BATCH_SIZE, gamma=GAMMA):
        if len(self.memory) < batch_size:
            self.t += 1
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
        next_state_batch = torch.zeros(batch_size) # will store the max future q_val of next_state
        with torch.no_grad():
            next_state_batch[non_terminal_states] = self.target_net(
                non_terminal_next_states).max(1)[0]
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        
       
        #reward_batch = torch.cat(batch.reward)
        reward_batch = torch.cat([torch.tensor([reward], dtype=torch.float32) for reward in batch.reward])

        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        
        # Compute the expected Q values
        expected_state_action_values = (next_state_batch * gamma) + reward_batch
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.t += 1



def train(env, agent, T):
    
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
            agent.eps -= EPSILON / (EPISODES * EXPLORE_FRAC) 
            agent.eps = max(agent.eps, MIN_EPS)
        else:
            next_state = next_state.flatten()
            next_state = torch.tensor([next_state], dtype=torch.float32)
            
        agent.memory.push(state, action, next_state, reward)
        agent.optimize()
        state = next_state
        
        
        
        if agent.t % UPDATE_EVERY == 0:
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            agent.target_net.load_state_dict(target_net_state_dict)
            
        if done: 
            break
        
    return episode_reward, info



df = pd.read_csv('gmedata.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Open', 'Close']].iloc[::-1]


env = gym.make('stocks-v0', df=df, frame_bound=(50,100), window_size=10)
input_dim = env.observation_space.shape[1] * env.observation_space.shape[0]
n_actions = env.action_space.n
T = env.unwrapped.frame_bound[1] - env.frame_bound[0]

agent = DQNAgent(input_dim, n_actions)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

x = []
y = []
for episode in range(EPISODES):
    reward, info = train(env, agent, T)
    print(f"Episode: {episode+1}, Reward: {reward}")
    print("info: ", info)
    print()
    x.append(episode + 1)
    y.append(reward)


x = np.array(x)  # Ensure x is a numpy array for consistent indexing
y = np.array(y)  # Ensure y is a numpy array

# Calculate the running average of the last 10 values
window_size = 20
weights = np.ones(window_size) / window_size
y_avg = np.convolve(y, weights, mode='valid')

# Plot the original rewards
plt.plot(x, y, label='Rewards')

# Plot the running average
# Note: The running average will have fewer points, adjust the x-axis accordingly
plt.plot(x[window_size - 1: ], y_avg, label='Running Average (Last 10)')

plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.grid()
plt.legend()
plt.show()




state, _ = env.reset()
agent.eps = 0
state = state.flatten()
state = torch.tensor([state], dtype=torch.float32)
while True:
    action = agent.policy_net(state).max(1)[1].view(1, 1)
    next_state, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated

    if done:
        print("info", info)
        break
    else:
        next_state = next_state.flatten()
        next_state = torch.tensor([next_state], dtype=torch.float32)
        state = next_state


plt.figure(figsize=(15,6))
plt.cla()
env.unwrapped.render_all()
plt.grid()
plt.show()

