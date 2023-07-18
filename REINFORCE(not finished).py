# import necessary packages
import gym
import numpy as np

env = gym.make("CartPole-v1")

def relu(mat):
    return np.multiply(mat, (mat > 0))

def relu_derivative(mat):
    return (mat > 0) * 1


class NNLayer:
    def __init__(self, input_size, output_size, activation=None, lr=0.001):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))  # initialize weights
        self.activation_function = activation

        self.lr = lr

    def forward(self, inputs, remember_for_backprop=True):
        input_with_bias = np.append(inputs, 1)
        unactivated = None
        unactivated = np.dot(input_with_bias, self.weights)
        output = unactivated
        if self.activation_function != None:
            output = self.activation_function(output)
        if remember_for_backprop:
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output

    def update_weights(self, gradient, gamma, cumulated_reward, t):
        self.weights += self.lr * (gamma ** t) * cumulated_reward * gradient


    def backward(self, gradient_from_above, gamma, cumulated_return, t):
        adjusted_mul = gradient_from_above
        if self.activation_function == None:
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out), gradient_from_above)
        D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))),
                     np.reshape(adjusted_mul, (1, len(adjusted_mul))))
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
        self.update_weights(D_i, gamma, cumulated_return, t)
        return delta_i


class RLAgent:
    env = None
    def __init__(self, env, num_hidden_layers=2, hidden_size=24, gamma=0.95):
        self.env = env
        self.hidden_size = hidden_size
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.num_hidden_layers = num_hidden_layers
        self.gamma = gamma
        self.layers = [
            NNLayer(self.input_size + 1, self.hidden_size, activation=relu),
            NNLayer(self.hidden_size + 1, self.hidden_size, activation=relu),
            NNLayer(self.hidden_size + 1, self.output_size)
        ]

    def select_action(self, observation):
        values = self.forward(np.asmatrix(observation), False)
        action_prob = np.exp(values - np.max(values)) / np.sum(np.exp(values - np.max(values)))
        if np.random.random() < action_prob[0]:
            return 0
        return 1

    def get_returns(self, rewards_list):
        returns = []
        G = 0
        for r in reversed(rewards_list):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def forward(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
            index = index + 1
        return vals

    def backward(self, observation, action, cumulated_return, t):
        values = self.forward(np.asmatrix(observation), True)
        delta = np.zeros_like(values)
        delta[action] = 1 / values[action]

        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.gamma, cumulated_return, t)




NUM_EPISODES = 10_000
MAX_TIMESTEPS = 1_000
GAMMA = 0.95
hidden_layer_size = 64
num_hidden_layers = 2

model = RLAgent(env, num_hidden_layers, hidden_layer_size, GAMMA)

for i_episode in range(NUM_EPISODES):
    rewards_list = []
    actions_list = []
    states_list = []
    observation = env.reset()
    states_list.append(observation)
    for t in range(MAX_TIMESTEPS):
        action = model.select_action(observation)
        prev_obs = observation
        observation, reward, done, info = env.step(action)
        rewards_list.append(reward)
        actions_list.append(action)
        states_list.append(observation)
        if done:
            print('Episode {} ended with total reward of {}'.format(i_episode, t))
            break

    G_t = model.get_returns(rewards_list)
    for i in range(len(G_t)):
        prev_obs, observation, action, reward = states_list[i], states_list[i + 1], actions_list[i], G_t[i]
        model.backward(prev_obs, action, reward, i + 1)


observation = env.reset()
for t in range(MAX_TIMESTEPS):
    action = model.select_action(observation)
    prev_obs = observation
    observation, reward, done, info = env.step(action)
    if done:
        break

print("Done")

