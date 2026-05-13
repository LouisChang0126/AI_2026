import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC, TD3, A2C, PPO

# 支援的演算法
ALGOS = {
    "sac": SAC,
    "td3": TD3,
    "a2c": A2C,
    "ppo": PPO,
}

def record_video(algo_name: str, model_path: str, video_length: int = 1000):
    algo_name = algo_name.lower()
    assert algo_name in ALGOS, f"不支援的演算法：{algo_name}"
    algo_class = ALGOS[algo_name]

    # 建立環境
    env = RecordVideo(
        gym.make("Ant-v5", render_mode="rgb_array"),
        video_folder="videos",
        name_prefix=f"{algo_name}_ant"
    )

    # 載入模型
    model = algo_class.load(model_path, env=env)

    obs, _ = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print("影片已儲存到 videos/ 資料夾。")

if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)
    # 範例使用 SAC 模型
    record_video(
        algo_name="sac",
        model_path="best_model_sac/best_model",
        video_length=1000
    )
