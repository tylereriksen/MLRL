import numpy as np
import gym

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden1, hidden2, lr=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr = lr

        self.weights1 = np.random.randn(self.input_size, self.hidden1) # (4, 64)
        self.bias1 = np.zeros((1, self.hidden1)) # (1, 64)
        self.weights2 = np.random.randn(self.hidden1, self.hidden2) # (64, 64)
        self.bias2 = np.zeros((1, self.hidden2)) # (1, 64)
        self.weights3 = np.random.randn(self.hidden2, self.output_size) # (64, 2)
        self.bias3 = np.zeros((1, self.output_size)) # (1, 2)

    def forward(self, x):

        def relu(x):
            return np.maximum(0, x)

        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3

        return self.z3

    def set_equal_parameters(self, otherNet):
        self.weights1 = np.copy(otherNet.weights1)
        self.bias1 = np.copy(otherNet.bias1)
        self.weights2 = np.copy(otherNet.weights2)
        self.bias2 = np.copy(otherNet.bias2)
        self.weights3 = np.copy(otherNet.weights3)
        self.bias3 = np.copy(otherNet.bias3)

    def backward(self, x, y_j, action_taken):
        def relu_derivative(i):
            return np.where(i > 0, 1, 0)

        self.forward(x)
        dL_dz3 = [[0, -2 * (y_j - self.z3[0][1])]] if action_taken == 1 else [[-2 * (y_j - self.z3[0][0]), 0]]
        dL_dw3 = np.dot(self.a2.T, dL_dz3) # multiply dz3_dw3 with dL_dz3
        dL_db3 = dL_dz3

        dL_dz2 = np.dot(dL_dz3, self.weights3.T) * relu_derivative(self.z2)
        dL_dw2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = dL_dz2

        dL_dz1 = np.dot(dL_dz2, self.weights2.T) * relu_derivative(self.z1)
        dL_dw1 = np.dot(x.T, dL_dz1)
        dL_db1 = dL_dz1

        self.weights3 -= self.lr * dL_dw3
        self.biases3 -= self.lr * dL_db3
        self.weights2 -= self.l1 * dL_dw2
        self.biases2 -= self.lr * dL_db2
        self.weights1 -= self.lr * dL_dw1
        self.biases1 -= self.lr * dL_db1



class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.count = 0

    def append(self, transition):
        if self.count < self.size:
            self.memory.append(transition)
        else:
            self.memory[self.count % self.size] = transition
        self.count += 1

    def getMemory(self):
        return self.memory



actionValFunc = NeuralNetwork(4, 2, 64, 64)
targetValFunc = NeuralNetwork(4, 2, 64, 64)
targetValFunc.set_equal_parameters(actionValFunc)
memory = ReplayMemory(10000)
EPSILON = 1.0
EPSILON_DECAY = 0.99
step_count = 0

env = gym.make("CartPole-v1")
for episode in range(10):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if np.random.rand() <= EPSILON:
            action = np.random.choice(2)
        else:
            action = np.argmax(actionValFunc.forward(state))
        EPSILON = np.maximum(EPSILON * EPSILON_DECAY, 0.05)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        memory.append((state, action, reward, next_state, done))

        y_j = reward if done else reward + np.max(targetValFunc.forward(next_state))
        actionValFunc.backward(state, y_j, action)

        state = next_state

        if step_count % 5 == 0:
            targetValFunc.set_equal_parameters(actionValFunc)
