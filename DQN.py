# import necessary packages
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt

# Constants
ENV_NAME = 'CartPole-v1'
REPLAY_MEMORY_SIZE = 10_000 # size of deque used for memory
MIN_REPLAY_MEMORY_SIZE = 1_000 # minimum amount of memory needed
BATCH_SIZE = 32
DISCOUNT = 0.99 # Ganna
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
TARGET_UPDATE_FREQ = 5
EPISODES = 100
graph_rewards = []


class DQNAgent:

    # initialize variables in this class
    # we will use two models to organize the .fit and .predict and to update in intervals
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self.__build_model() # .fitting every step

        self.target_model = self.__build_model() # .predict every step
        self.target_model.set_weights(self.model.get_weights())

        self.target_update_counter = 0

    # the Neural net to build our model
    def __build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    # adding transitions to memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # choosing actions based on epsilon and q-values
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(np.reshape(state, (1, self.state_size)))
        return np.argmax(q_values[0])

    # we will train the model on the observations here that are in memory
    def replay(self, batch_size, terminal_state):

        # if less than necessary size, we will not do anything to the model
        if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # select some transitions from memory randmly
        minibatch = random.sample(self.memory, batch_size)

        # save the current and next states, along with the q-values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        # places to store the states and q-values
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))

        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward  # target is the new q_value

            if not done:
                next_q_values = np.max(future_qs_list[idx])
                target += DISCOUNT * next_q_values
            else:
                target += 0

            # get the q-values and update the one with the action taken
            q_values = current_qs_list[idx]
            q_values[action] = target

            # store it in the variables
            states[idx] = state
            targets[idx] = q_values

        self.model.fit(states, targets, batch_size=batch_size, epochs=10, verbose=0, shuffle=False)

        if terminal_state is True:
            self.target_update_counter += 1

        # update in intervals
        if self.target_update_counter % TARGET_UPDATE_FREQ == 0:
            self.target_model.set_weights(self.model.get_weights())

        # update epsilon accordingly
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY



# Create the environment
env = gym.make(ENV_NAME)

# Observation Space: cart position, cart velocity, pole angle, pole angular velocity
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(env.observation_space.high) # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low) # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

agent = DQNAgent(state_size, action_size) # create learning agent

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        agent.add(state, action, reward, next_state, done)
        agent.replay(BATCH_SIZE, done)
        state = next_state

    print(f"Episode: {episode + 1} / Total Reward: {total_reward}")
    graph_rewards.append(total_reward)


plt.plot(range(len(graph_rewards)), graph_rewards, 'r-')
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards Collected Over Episodes")
plt.show()

# Test the trained agent
state = env.reset()
total_reward = 0

while True:
    env.render()
    action = agent.act(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward

    if done:
        print(f"Total Reward: {total_reward}")
        break

env.close()



