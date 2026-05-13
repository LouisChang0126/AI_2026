import os
import argparse
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from tqdm import tqdm
import numpy as np
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TOTAL_TIMESTEPS = 20_000_000

class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps, save_interval=10, use_clip_reward=False, algorithm="PPO"):
        super().__init__()
        self.rewards = []
        self.ep_lengths = []
        self.total_timesteps = total_timesteps
        self.pbar = tqdm(total=total_timesteps)
        self.save_interval = save_interval
        self.episode_count = 0
        self.use_clip_reward = use_clip_reward
        self.algorithm = algorithm

    def _on_step(self) -> bool:
        self.pbar.update(self.model.n_envs)
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                ep_info = info['episode']
                self.rewards.append(ep_info['r'])
                self.ep_lengths.append(ep_info['l'])
                self.episode_count += 1
                if self.episode_count % self.save_interval == 0:
                    self._plot_rewards()
        return True

    def _on_training_end(self) -> None:
        self._plot_rewards()
        self.pbar.close()
    
    # reward 曲線
    def _plot_rewards(self):
        if not self.rewards:
            return
        plt.figure()
        plt.plot(self.rewards, color="blue", label="Reward")
        window = 50
        if len(self.rewards) >= window:
            avg_rewards = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window - 1, len(self.rewards)), avg_rewards, color="red", label="Avg Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"breakout-v5 Training Reward ({self.algorithm.upper()}) env:{self.model.n_envs}{' (with clip)' if self.use_clip_reward else ''}")
        plt.legend()
        plt.grid()
        plt.savefig(f"reward_plot/breakout_reward_curve_{self.algorithm.lower()}_env{self.model.n_envs}{'_clip' if self.use_clip_reward else ''}.png")
        plt.close()

def make_env(rank=0, use_clip_reward=False):
    def _init():
        env = gym.make("ALE/Breakout-v5")
        if use_clip_reward:
            print("use ClipRewardEnv")
            env = ClipRewardEnv(env)
        env.reset(seed=100 + rank)
        return Monitor(env)
    return _init

def main():
    os.makedirs("reward_plot", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, required=True, help="要建立的環境數量 (for VecEnv)")
    parser.add_argument("--algo", type=str, required=True, choices=["A2C", "PPO"], help="選擇使用的演算法：A2C 或 PPO")
    parser.add_argument('--clip', action='store_true', help='使用 ClipRewardEnv')
    args = parser.parse_args()

    use_clip_reward = args.clip
    algorithm = args.algo.upper()
    env_fns = [make_env(rank, use_clip_reward) for rank in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    eval_env = Monitor(gym.make("ALE/Breakout-v5"))

    if algorithm == "PPO":
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=256 // args.num_envs,  # 確保 batch size 不會太大，可依實際需求調整
            batch_size=32,
            n_epochs=4,
            ent_coef=0.01,
            clip_range=0.1,
            verbose=1,
            device="cuda"
        )
    else:
        model = A2C(
            "CnnPolicy",
            env,
            learning_rate=7e-4,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            device="cuda"
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_model_{algorithm.lower()}{'_clip' if use_clip_reward else ''}",
        log_path=f"./eval_logs_{algorithm.lower()}{'_clip' if use_clip_reward else ''}",
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    reward_logger = TQDMCallback(TOTAL_TIMESTEPS, save_interval=10, use_clip_reward=use_clip_reward, algorithm=algorithm)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[reward_logger, eval_callback]
    )

    # 儲存最終模型
    os.makedirs("final_models", exist_ok=True)
    model.save(f"final_models/{algorithm.lower()}_breakout_final{'_clip' if use_clip_reward else ''}")

    # 儲存最終的 training_log 曲線
    os.makedirs("training_log", exist_ok=True)
    with open(f"training_log/training_log_{algorithm.lower()}{'_clip' if use_clip_reward else ''}.pkl", "wb") as f:
        pickle.dump({
            "rewards": reward_logger.rewards,
        }, f)

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
