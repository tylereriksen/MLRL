import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import gym_anytrading


# Load the dataset
df = pd.read_csv('gmedata.csv')

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Open', 'Close']].iloc[::-1]

# Create and wrap the environment
env = gym.make('stocks-v0', df=df, frame_bound=(50,100), window_size=10)


# Evaluate the trained agent
obs, _ = env.reset()
total_profit = 0
total_reward = 0
total_position = 0
while True:
    action = 2
    obs, rewards, term, trun, info = env.step(action)
    total_profit += info['total_profit']
    total_reward += info['total_reward']
    # total_position += 1 if info['position'] == gym_anytrading.envs.Positions.Long else -1

    done = term or trun
    if done:
        print("info:", info)
        break

# Render the environment to visualize the agent's performance
plt.figure(figsize=(15,6))
#plt.cla()
plt.title(f"Total Reward: {total_reward} ~ Total Profit: {total_profit}")
env.render_all()
plt.show()