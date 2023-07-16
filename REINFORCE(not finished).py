import gym
import numpy as np

LEARNING_RATE = 0.001
EPISODES = 300
HIDDEN_LAYER1 = 64
HIDDEN_LAYER2 = 64
DISCOUNT = 0.99
LEARNING_RATE = 0.001
i = [2, 3]

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden1, hidden2, lr=LEARNING_RATE):
        self.input_size = input_size  # input size
        self.output_size = output_size  # output size
        self.hidden1 = hidden1  # hidden layer 1 size
        self.hidden2 = hidden2  # hidden layer 2 size
        self.lr = lr  # learning rate

        self.weights1 = np.random.randn(self.input_size, self.hidden1) * np.sqrt(
            2 / (self.input_size + self.output_size))  # (4, 64)
        self.bias1 = np.zeros((1, self.hidden1))  # (1, 64)
        self.weights2 = np.random.randn(self.hidden1, self.hidden2) * np.sqrt(
            2 / (self.input_size + self.output_size))  # (64, 64)
        self.bias2 = np.zeros((1, self.hidden2))  # (1, 64)
        self.weights3 = np.random.randn(self.hidden2, self.output_size) * np.sqrt(
            2 / (self.input_size + self.output_size))  # (64, 2)
        self.bias3 = np.zeros((1, self.output_size))  # (1, 2)

        self.z1, self.a1, self.z2, self.a2, self.z3, self.action_probs = None, None, None, None, None, None


    def forward(self, x):
        # define rectified linear function
        def relu(i):
            return np.maximum(0, i)

        self.z1 = np.dot(x, self.weights1) + self.bias1 # (1, 64)
        self.a1 = relu(self.z1) # (1, 64)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2 # (1, 64)
        self.a2 = relu(self.z2) # (1, 64)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3 # (1, 2)

        self.action_probs = np.exp(self.z3) / np.sum(np.exp(self.z3), axis=1, keepdims=True)

        return self.action_probs

    def select_action(self, probs):
        r = np.random.rand()
        if r < probs[0]:
            return 0
        return 1

    def calculate_gradient_log_probs(self, x, a):
        action_probs = self.forward(x)

        # Calculate ∇a log π(a|s)
        grad_log_probs = np.zeros_like(self.action_probs)
        grad_log_probs[0, a] = 1 / self.action_probs[0, a]

        # Backpropagation to calculate ∇θ a
        grad_a3 = grad_log_probs
        grad_weights3 = np.dot(self.a2.T, grad_a3)
        grad_bias3 = np.sum(grad_a3, axis=0, keepdims=True)
        grad_a2 = np.dot(grad_a3, self.weights3.T)
        grad_z2 = np.multiply(grad_a2, np.where(self.z2 > 0, 1, 0))
        grad_weights2 = np.dot(self.a1.T, grad_z2)
        grad_bias2 = np.sum(grad_z2, axis=0, keepdims=True)
        grad_a1 = np.dot(grad_z2, self.weights2.T)
        grad_z1 = np.multiply(grad_a1, np.where(self.z1 > 0, 1, 0))
        grad_weights1 = np.dot(x.T, grad_z1)
        grad_bias1 = np.sum(grad_z1, axis=0, keepdims=True)

        # Gradients ∇θ log π(a|s)
        gradients = {
            'weights1': grad_weights1,
            'bias1': grad_bias1,
            'weights2': grad_weights2,
            'bias2': grad_bias2,
            'weights3': grad_weights3,
            'bias3': grad_bias3
        }

        return self.z3, gradients


def calculate_cumulative_rewards(rewards):
    G_t = [rewards[-1]]
    idx = len(rewards) - 2
    while idx >= 0:
        G_t.insert(0, G_t[0] * DISCOUNT + rewards[idx])
        idx -= 1
    return np.array(G_t)


# Create the environment
env = gym.make("CartPole-v1")

# Observation Space: cart position, cart velocity, pole angle, pole angular velocity
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(env.observation_space.high) # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low) # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]


net = NeuralNetwork(state_size, action_size, HIDDEN_LAYER1, HIDDEN_LAYER2)


for episode in range(EPISODES):
    state = env.reset()
    done = False

    states = [state]
    actions = []
    rewards = []
    z3s = []
    gradients = []

    while not done:
        action_prob = net.forward(state)
        action = net.select_action(action_prob[0])
        next_state, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        z3, gradient = net.calculate_gradient_log_probs(np.array([state]), action)
        z3s.append(z3)
        gradients.append(gradient)

        state = next_state

    if episode == 0:
        print(gradients[-1]["weights1"])
    if episode == EPISODES - 1:
        print(gradients[-1]["weights1"])

    cumulated_rewards = calculate_cumulative_rewards(rewards)
    for idx in range(len(cumulated_rewards)):
        net.weights1 += LEARNING_RATE * DISCOUNT ** (idx + 1) * cumulated_rewards[idx] * gradients[idx]["weights1"]
        net.bias1 += LEARNING_RATE * DISCOUNT ** (idx + 1) * cumulated_rewards[idx] * gradients[idx]["bias1"]
        net.weights2 += LEARNING_RATE * DISCOUNT ** (idx + 1) * cumulated_rewards[idx] * gradients[idx]["weights2"]
        net.bias2 += LEARNING_RATE * DISCOUNT ** (idx + 1) * cumulated_rewards[idx] * gradients[idx]["bias2"]
        net.weights3 += LEARNING_RATE * DISCOUNT ** (idx + 1) * cumulated_rewards[idx] * gradients[idx]["weights3"]
        net.bias3 += LEARNING_RATE * DISCOUNT ** (idx + 1) * cumulated_rewards[idx] * gradients[idx]["bias3"]




