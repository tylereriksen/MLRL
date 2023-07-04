import numpy as np
import gym
import random

EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.05
MIN_REPLAY_MEMORY_SIZE = 1_000
REPLAY_MEMORY_SIZE = 10_000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQUENCY = 5
EPISODES = 100
HIDDEN_LAYER1 = 64
HIDDEN_LAYER2 = 64


class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden1, hidden2, lr=LEARNING_RATE):
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

        def relu(i):
            return np.maximum(0, i)

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
        dL_db3 = np.array(dL_dz3)

        dL_dz2 = np.dot(dL_dz3, self.weights3.T) * relu_derivative(self.z2)
        dL_dw2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = dL_dz2

        dL_dz1 = np.dot(dL_dz2, self.weights2.T) * relu_derivative(self.z1)
        dL_dw1 = np.dot(x.T, dL_dz1)
        dL_db1 = dL_dz1

               # 64-2,
        return [dL_dw3, dL_db3, dL_dw2, dL_db2, dL_dw1, dL_db1]

    def update_parameters(self, dw3, db3, dw2, db2, dw1, db1):
        self.weights3 -= self.lr * dw3
        self.bias3 -= self.lr * db3
        self.weights2 -= self.lr * dw2
        self.bias2 -= self.lr * db2
        self.weights1 -= self.lr * dw1
        self.bias1 -= self.lr * db1


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

    def returnBatch(self, batch_size):
        if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
            return None
        minibatch = random.sample(self.memory, batch_size)
        return minibatch


# Create the environment
env = gym.make("CartPole-v1")

# Observation Space: cart position, cart velocity, pole angle, pole angular velocity
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(env.observation_space.high) # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low) # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

actionValFunc = NeuralNetwork(state_size, action_size, HIDDEN_LAYER1, HIDDEN_LAYER2)
targetValFunc = NeuralNetwork(state_size, action_size, HIDDEN_LAYER1, HIDDEN_LAYER2)
targetValFunc.set_equal_parameters(actionValFunc)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

step_count = 0
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        # get action
        if np.random.rand() <= EPSILON:
            action = np.random.choice(action_size)
        else:
            action = np.argmax(actionValFunc.forward(state))
        # update epsilon
        EPSILON = np.maximum(EPSILON * EPSILON_DECAY, EPSILON_MIN)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        memory.append((state, action, reward, next_state, done))

        minibatch = memory.returnBatch(BATCH_SIZE)

        if minibatch is None:
            step_count += 1
            state = next_state
            continue

        gradients = {"w3": [], "b3": [], "w2": [], "b2": [], "w1": [], "b1" : []}
        for transition in minibatch:
            y_j = transition[2] if transition[4] else transition[2] + np.max(targetValFunc.forward(transition[3]))
            gradient = actionValFunc.backward(np.array([transition[0]]), y_j, transition[1])
            gradients["w3"].append(gradient[0].tolist())
            gradients["b3"].append(gradient[1].tolist())
            gradients["w2"].append(gradient[2].tolist())
            gradients["b2"].append(gradient[3].tolist())
            gradients["w1"].append(gradient[4].tolist())
            gradients["b1"].append(gradient[5].tolist())

        dw3, db3 = np.mean(np.array(gradients["w3"]), axis=0), np.mean(np.array(gradients["b3"]), axis=0)
        dw2, db2 = np.mean(np.array(gradients["w2"]), axis=0), np.mean(np.array(gradients["b2"]), axis=0)
        dw1, db1 = np.mean(np.array(gradients["w1"]), axis=0), np.mean(np.array(gradients["b1"]), axis=0)

        actionValFunc.update_parameters(dw3, db3, dw2, db2, dw1, db1)

        state = next_state

        if step_count % TARGET_UPDATE_FREQUENCY == 0:
            targetValFunc.set_equal_parameters(actionValFunc)

        step_count += 1


env.close()
