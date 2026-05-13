import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# 畫圖
plt.figure()

# 讀取 a2c 資料
with open("training_log/training_log_a2c.pkl", "rb") as f:
    data = pickle.load(f)
a2c_rewards = data["rewards"]

# 讀取 a2c_clip 資料
with open("training_log/training_log_a2c_clip.pkl", "rb") as f:
    data2 = pickle.load(f)
a2c_clip_rewards = data2["rewards"]

# 讀取 ppo 資料
with open("training_log/training_log_ppo.pkl", "rb") as f:
    data = pickle.load(f)
ppo_rewards = data["rewards"]

# 讀取 ppo_clip 資料
with open("training_log/training_log_ppo_clip.pkl", "rb") as f:
    data2 = pickle.load(f)
ppo_clip_rewards = data2["rewards"]


# 移動平均 (例如 window=50)
window = 2000
if len(a2c_rewards) >= window:
    avg_rewards = np.convolve(a2c_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(a2c_rewards)), avg_rewards, label="A2C Reward", color="red")
    
if len(a2c_clip_rewards) >= window:
    clip_avg_rewards = np.convolve(a2c_clip_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(a2c_clip_rewards)), clip_avg_rewards, label="A2C Clip Reward", color="blue")

if len(ppo_rewards) >= window:
    avg_rewards = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(ppo_rewards)), avg_rewards, label="PPO Reward", color="orange")
    
if len(ppo_clip_rewards) >= window:
    clip_avg_rewards = np.convolve(ppo_clip_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(ppo_clip_rewards)), clip_avg_rewards, label="PPO Clip Reward", color="green")

os.makedirs("reward_plot", exist_ok=True)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Breakout-v5 Average Reward Curves (window=2000)")
plt.legend()
plt.grid()
plt.savefig("reward_plot/breakout_reward_curves.png")
