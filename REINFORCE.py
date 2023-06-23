import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

'''
This algorithm is highly naiive so there is a lot of variance depending on how it is trained
and the trajectories produced
'''


ENV_NAME = "CartPole-v1"
LAYER1 = 64
LEARNING_RATE = 0.001
EPISODES = 20000
DISCOUNT = 0.99


# Define the policy network
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, LAYER1)
        self.fc2 = nn.Linear(LAYER1, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)  # apply softmax activation to obtain action probabilities


# Function to collect trajectories
def collect_trajectories(policy, env, num_episodes):
    trajectories = []
    for _ in range(num_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        while not done:
            states.append(state)
            state_tensor = torch.FloatTensor(state)
            action_probs = policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)  # create distribution of actions
            action = action_dist.sample()  # sample an action from the distribution
            actions.append(action)
            state, reward, done, _ = env.step(action.item())  # take the chosen action in the environment
            rewards.append(reward)
        trajectory = {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'rewards': torch.FloatTensor(rewards)
        }
        trajectories.append(trajectory)
    return trajectories


# Function to compute policy loss
def compute_loss(policy, states, actions, returns):
    action_probs = policy(states)  # compute action probabilities using the policy
    action_dist = torch.distributions.Categorical(action_probs)  # create a distribution over actions
    log_probs = action_dist.log_prob(actions)  # compute the log probabilities of the chosen actions
    loss = -(log_probs * returns).mean()  # compute the policy loss
    return loss


# Function to update the policy
def update_policy(policy, optimizer, trajectories, discount_factor):
    for trajectory in trajectories:
        states = trajectory['states'] # get the states from monte carlo simulations
        actions = trajectory['actions'] # get the actions from monte carlo simulations
        rewards = trajectory['rewards'] # get the rewards from monte carlo simulations

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + discount_factor * G  # compute the discounted return
            returns.insert(0, G)  # insert the return at the beginning of the returns list
        returns = torch.FloatTensor(returns)

        optimizer.zero_grad()  # zero the gradients of the optimizer
        loss = compute_loss(policy, states, actions, returns)  # compute the policy loss
        loss.backward()  # backpropagate the gradients
        optimizer.step()  # update the policy parameters using the optimizer



# Create the environment
env = gym.make(ENV_NAME)

# Observation Space: cart position, cart velocity, pole angle, pole angular velocity
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(env.observation_space.high) # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low) # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]


policy = Policy(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# Collect trajectories and update policy
trajectories = collect_trajectories(policy, env, EPISODES)
update_policy(policy, optimizer, trajectories, DISCOUNT)

trajectories = None

# Test the learned policy
state = env.reset()
done = False
total_reward = 0
for _ in range(10):
    while not done:
        env.render()
        state_tensor = torch.FloatTensor(state)
        action_probs = policy(state_tensor)
        action = torch.argmax(action_probs).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print("Total reward:", total_reward)
    state = env.reset()
    total_reward = 0
    done = False

# Close the environment
env.close()
