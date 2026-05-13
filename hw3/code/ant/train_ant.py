import os
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, TD3, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from tqdm import tqdm
import pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TOTAL_TIMESTEPS = 1_000_000

# 支援演算法選擇
ALGOS = {
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
    "PPO": PPO,
}

# tqdm callback
class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.rewards = []
        self.ep_lengths = []
        self.total_timesteps = total_timesteps
        self.pbar = tqdm(total=total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(self.model.n_envs)
        if 'episode' in self.locals.get('infos', [{}])[0]:
            ep_info = self.locals['infos'][0]['episode']
            self.rewards.append(ep_info['r'])
            self.ep_lengths.append(ep_info['l'])
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="SAC", choices=ALGOS.keys(),
                        help="選擇使用的演算法：SAC、TD3、A2C 或 PPO")
    args = parser.parse_args()

    algo_name = args.algo.upper()
    algo_class = ALGOS[algo_name]

    # 建立環境
    env = Monitor(gym.make("Ant-v5", render_mode="rgb_array"))
    eval_env = Monitor(gym.make("Ant-v5", render_mode="rgb_array"))

    # 建立模型
    model = algo_class(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=3e-4
    )

    # 評估 callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_model_{algo_name.lower()}",
        log_path=f"./eval_logs_{algo_name.lower()}",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    # 訓練
    reward_logger = TQDMCallback(TOTAL_TIMESTEPS)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[reward_logger, eval_callback]
    )

    # 儲存最終模型
    os.makedirs("final_models", exist_ok=True)
    model.save(f"final_models/{algo_name.lower()}_ant_final")

    # 儲存最終的 training_log 曲線
    os.makedirs("training_log", exist_ok=True)
    with open(f"training_log/training_log_{algo_name.lower()}.pkl", "wb") as f:
        pickle.dump({
            "rewards": reward_logger.rewards,
        }, f)

    # reward 曲線
    os.makedirs("reward_plot", exist_ok=True)
    plt.figure()
    plt.plot(reward_logger.rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    plt.title(f"Ant-v5 Training Reward ({algo_name})")
    plt.grid()
    plt.savefig(f"reward_plot/ant_reward_curve_{algo_name.lower()}.png")
    plt.show()

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
