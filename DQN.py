# import necessary packages
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

# define the activation function and its derivative
def relu(mat):
    return np.multiply(mat, (mat > 0))


def relu_derivative(mat):
    return (mat > 0) * 1


# class representing a neural net layer
class NNLayer:
    def __init__(self, input_size, output_size, activation=None, lr=0.001):
        self.input_size = input_size
        self.output_size = output_size

        # weights of network that we update every step (action-value function Q)
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size)) # initialize weights
        # target action-value function Q-hat
        self.stored_weights = np.copy(self.weights) # target network: updated every so often

        self.activation_function = activation

        # for gradient descent optimization with Adam
        self.lr = lr
        self.m = np.zeros((input_size, output_size))
        self.v = np.zeros((input_size, output_size))
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.time = 1
        self.adam_epsilon = 0.00000001

    # Compute the forward pass for this layer
    def forward(self, inputs, remember_for_backprop=True):
        # inputs has shape batch_size x layer_input_size
        input_with_bias = np.append(inputs, 1)
        unactivated = None
        if remember_for_backprop:
            unactivated = np.dot(input_with_bias, self.weights)
        else:
            unactivated = np.dot(input_with_bias, self.stored_weights)
        # store variables for backward pass
        output = unactivated
        if self.activation_function != None:
            # assuming here the activation function is relu, this can be made more robust
            output = self.activation_function(output)
        if remember_for_backprop:
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output

    def update_weights(self, gradient):
        m_temp = np.copy(self.m)
        v_temp = np.copy(self.v)

        m_temp = self.beta_1 * m_temp + (1 - self.beta_1) * gradient
        v_temp = self.beta_2 * v_temp + (1 - self.beta_2) * (gradient * gradient)
        m_vec_hat = m_temp / (1 - np.power(self.beta_1, self.time + 0.1))
        v_vec_hat = v_temp / (1 - np.power(self.beta_2, self.time + 0.1))
        self.weights = self.weights - np.divide(self.lr * m_vec_hat, np.sqrt(v_vec_hat) + self.adam_epsilon)

        self.m = np.copy(m_temp)
        self.v = np.copy(v_temp)

    def update_stored_weights(self):
        self.stored_weights = np.copy(self.weights)

    def update_time(self):
        self.time = self.time + 1

    def backward(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        # this is pointwise
        if self.activation_function != None:
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out), gradient_from_above)
        D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))),
                     np.reshape(adjusted_mul, (1, len(adjusted_mul))))
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
        self.update_weights(D_i)
        return delta_i


class RLAgent:
    # class representing a reinforcement learning agent
    env = None

    def __init__(self, env, num_hidden_layers=2, hidden_size=24, gamma=0.95, epsilon_decay=0.997, epsilon_min=0.01):
        self.env = env
        self.hidden_size = hidden_size
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.num_hidden_layers = num_hidden_layers
        self.epsilon = 1.0 # STARTING EPSILON
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque([], 1000000)
        self.gamma = gamma
        self.time = 1
        self.update_every = 5

        self.layers = [
            NNLayer(self.input_size + 1, self.hidden_size, activation=relu),
            NNLayer(self.hidden_size + 1, self.hidden_size, activation=relu),
            NNLayer(self.hidden_size + 1, self.output_size)
        ]

    def select_action(self, observation):
        values = self.forward(np.asmatrix(observation))
        if (np.random.random() > self.epsilon):
            return np.argmax(values)
        else:
            return np.random.randint(self.env.action_space.n)

    def select_action_for_benchmark(self, observation):
        values = self.forward(np.asmatrix(observation))
        return np.argmax(values)

    def forward(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
            index = index + 1
        return vals

    def remember(self, done, action, observation, prev_obs):
        self.memory.append([done, action, observation, prev_obs])

    def experience_replay(self, update_size=20):
        if (len(self.memory) < update_size):
            return
        else:
            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, new_obs, prev_obs = self.memory[index]
                action_values = self.forward(prev_obs, remember_for_backprop=True)
                next_action_values = self.forward(new_obs, remember_for_backprop=False)
                experimental_values = np.copy(action_values)
                if done:
                    experimental_values[action_selected] = -1
                else:
                    experimental_values[action_selected] = 1 + self.gamma * np.max(next_action_values)
                self.backward(action_values, experimental_values)
        self.epsilon = self.epsilon if self.epsilon < self.epsilon_min else self.epsilon * self.epsilon_decay
        for layer in self.layers:
            layer.update_time()
            if self.time % self.update_every == 0:
                layer.update_stored_weights()
            self.time += 1

    def backward(self, calculated_values, experimental_values):
        # values are batched = batch_size x output_size
        delta = 2 * (calculated_values - experimental_values) # Gradient of Loss Function

        for layer in reversed(self.layers):
            delta = layer.backward(delta)




# Global variables
NUM_EPISODES = 10_000 # NUMBER OF EPISODES
MAX_TIMESTEPS = 1_000 # MAXIMUM NUMBERS OF STEPS PER EPISODE
AVERAGE_REWARD_TO_SOLVE = 200 # AVERAGE REWARD PER EPISODE NEEDED TO CONSIDER ENVIRONNMENT SOLVED
NUM_EPS_TO_SOLVE = 100 # THE ROLLING AVERAGE OF THE LAST NUMBER OF EPISODES THAT WE WILL LOOK AT
GAMMA = 0.95 # DISCOUNT
EPSILON_DECAY = 0.997 # DECAY RATIO OF EPSILON-GREEDY
update_size = 20 # REPLAY BUFFER SIZE
hidden_layer_size = 64 # SIZE OF HIDDEN LAYERS
num_hidden_layers = 2 # NUMBER OF HIDDEN LAYERS IN NEURAL NET

model = RLAgent(env, num_hidden_layers, hidden_layer_size, GAMMA, EPSILON_DECAY)
scores_last_timesteps = deque([], NUM_EPS_TO_SOLVE)


# benchmark rewards
train_rewards = []
for i_episode in range(100):
    observation = env.reset()
    total_rewards = 0
    for t in range(MAX_TIMESTEPS):
        action = model.select_action_for_benchmark(observation)
        prev_obs = observation
        observation, reward, done, info = env.step(action)
        total_rewards += reward
        if done:
            # If the pole has tipped over, end this episode
            print('Episode {} ended with total reward of {}'.format(i_episode, total_rewards))
            scores_last_timesteps.append(t + 1)
            train_rewards.append(total_rewards)
            break
print('Average reward before training was {}'.format(np.mean(np.array(train_rewards))))



# The main program loop
test_rewards = []
for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    rewards = []
    total_rewards = 0
    if i_episode >= NUM_EPS_TO_SOLVE:
        if (sum(scores_last_timesteps) / NUM_EPS_TO_SOLVE > AVERAGE_REWARD_TO_SOLVE):
            print("solved after {} episodes".format(i_episode))
            break
    # Iterating through time steps within an episode
    for t in range(MAX_TIMESTEPS):
        # env.render()
        action = model.select_action(observation)
        prev_obs = observation
        observation, reward, done, info = env.step(action)
        total_rewards += reward
        rewards.append(reward)
        # Keep a store of the agent's experiences
        model.remember(done, action, observation, prev_obs)
        model.experience_replay(update_size)
        # epsilon decay
        if done:
            # If the pole has tipped over, end this episode
            print('Episode {} ended with a mean reward of {}'.format(i_episode, len(rewards)))
            scores_last_timesteps.append(t + 1)
            break

    test_rewards.append(total_rewards)


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
