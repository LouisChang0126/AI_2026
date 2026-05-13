import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

# 支援演算法選擇
ALGOS = {
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
    "PPO": PPO,
}

def test_model(model_path: str, algo_name: str, episodes: int = 5):
    algo_class = ALGOS[algo_name.upper()]
    env = Monitor(gym.make("Ant-v5"))

    model = algo_class.load(model_path, env=env, device="cpu")

    rewards = 0.0
    for ep in tqdm(range(episodes)):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards += total_reward
        # print(f"Episode {ep + 1}: Reward = {total_reward}")
    print(f"{algo_name.upper()} Average Reward: {rewards / episodes}")
    env.close()


if __name__ == "__main__":
    test_model("best_model_a2c/best_model", "a2c", episodes=50)
    test_model("best_model_ppo/best_model", "ppo", episodes=50)
    test_model("best_model_sac/best_model", "sac", episodes=50)
    test_model("best_model_td3/best_model", "td3", episodes=50)