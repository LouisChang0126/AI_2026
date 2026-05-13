import os
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C, PPO

# 支援的演算法
ALGOS = {
    "a2c": A2C,
    "ppo": PPO,
}

def record_video(algo_name: str, model_path: str, video_length: int = 1000):
    algo_name = algo_name.lower()
    assert algo_name in ALGOS, f"不支援的演算法：{algo_name}"
    algo_class = ALGOS[algo_name]

    # 建立環境
    env = RecordVideo(
        gym.make("ALE/Breakout-v5", render_mode="rgb_array"),
        video_folder="videos",
        name_prefix=f"{algo_name}_breakout"
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
    record_video(
        algo_name="ppo",
        model_path="best_model_ppo_clip/best_model",
        video_length=10000
    )
