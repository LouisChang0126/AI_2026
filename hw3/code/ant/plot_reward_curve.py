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

# 讀取 sac 資料
with open("training_log/training_log_sac.pkl", "rb") as f:
    data = pickle.load(f)
sac_rewards = data["rewards"]

# 讀取 ppo 資料
with open("training_log/training_log_ppo.pkl", "rb") as f:
    data = pickle.load(f)
ppo_rewards = data["rewards"]

# 讀取 td3 資料
with open("training_log/training_log_td3.pkl", "rb") as f:
    data = pickle.load(f)
td3_rewards = data["rewards"]


# 移動平均 (例如 window=50)
window = 200
if len(a2c_rewards) >= window:
    avg_rewards = np.convolve(a2c_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(a2c_rewards)), avg_rewards, label="A2C Reward", color="red")
    
if len(sac_rewards) >= window:
    avg_rewards = np.convolve(sac_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(sac_rewards)), avg_rewards, label="SAC Reward", color="blue")

if len(ppo_rewards) >= window:
    avg_rewards = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(ppo_rewards)), avg_rewards, label="PPO Reward", color="orange")
    
if len(td3_rewards) >= window:
    avg_rewards = np.convolve(td3_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(td3_rewards)), avg_rewards, label="TD3 Reward", color="green")


os.makedirs("reward_plot", exist_ok=True)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Ant-v5 Average Reward Curves (window=200)")
plt.legend()
plt.grid()
plt.savefig("reward_plot/ant_reward_curves.png")
