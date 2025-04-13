import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import pandas as pd
import os
from evaluate import makeVideo

# Sparse reward wrapper
class SparseLunarRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            if reward > 0:

                reward = 100
            else:
                reward = -10
        else:
            reward = -10

        return obs, reward, terminated, truncated, info

# 設定訓練參數
TOTAL_TIMESTEPS = 1_000_000
REWARD_MODES = {
    "dense": None,
    "sparse": SparseLunarRewardWrapper
}

# # 執行訓練與紀錄
# results = {}

# for mode, wrapper in REWARD_MODES.items():
#     print(f"\n訓練模式：{mode.upper()} REWARD")
#     log_dir = f"./logs/ppo_reward_{mode}/"
#     os.makedirs(log_dir, exist_ok=True)

#     def make_env():
#         env = gym.make("LunarLander-v2")
#         if wrapper:
#             env = wrapper(env)
#         env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
#         return env

#     env = DummyVecEnv([make_env])

#     model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=TOTAL_TIMESTEPS)

#     # 評估模型
#     eval_env = gym.make("LunarLander-v2")
#     if wrapper:
#         eval_env = wrapper(eval_env)
#     mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
#     print(f"{mode.upper()} - Avg Reward：{mean_reward:.2f} ± {std_reward:.2f}")

#     results[mode] = {
#         "mean_reward": mean_reward,
#         "std_reward": std_reward
#     }
#     makeVideo(model, "LunarLander-v2", filename=f"lunarlander_ppo_{mode}.gif")

# 畫圖 reward trend（使用 monitor log）
plt.figure(figsize=(10, 6))
for mode in REWARD_MODES:
    log_path = f"./logs/ppo_reward_{mode}/monitor.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path, skiprows=1)
        rewards = df["r"].rolling(window=10).mean()
        plt.plot(rewards, label=f"{mode} reward")

plt.title("Reward Trend: Dense vs Sparse")
plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_dense_vs_sparse.png")
plt.show()

print("\n實驗完成！結果儲存在 ./logs 與 reward_dense_vs_sparse.png")
