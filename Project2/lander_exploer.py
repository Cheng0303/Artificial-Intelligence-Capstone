import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from evaluate import makeVideo

import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Entropy 實驗設定（4 組）
entropy_settings = {
    "0.0": 0.0,
    "0.01": 0.01,
    "0.1": 0.1,
    "0.5": 0.5
}

# 執行每組訓練
for setting_name, entropy_coef in entropy_settings.items():
    print(f"\n訓練模型：entropy_coef = {entropy_coef}")

    log_dir = f"./logs/ppo_entropy_{setting_name}/"
    os.makedirs(log_dir, exist_ok=True)

    # 建立環境 + Monitor 記錄 log
    def make_env():
        env = gym.make("LunarLander-v2")
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        return env

    env = DummyVecEnv([make_env])

    # 建立模型，設定探索程度
    model = PPO("MlpPolicy", env,
                verbose=1,
                learning_rate=2.5e-4,
                n_steps=2048,
                batch_size=64,
                ent_coef=entropy_coef
                )

    # 設定 tensorboard logger（可選）
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # 訓練模型
    model.learn(total_timesteps=1_000_000)

    # 儲存模型
    model_path = f"./models/ppo_entropy_{setting_name}"
    os.makedirs("./models", exist_ok=True)
    model.save(model_path)

    # 評估模型
    eval_env = gym.make("LunarLander-v2")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Avg Reward（entropy={entropy_coef}）：{mean_reward:.2f} ± {std_reward:.2f}")

    # 錄影模型行為
    time.sleep(3)
    makeVideo(model, "LunarLander-v2", filename=f"lunarlander_ppo_entropy_{setting_name}")

# 畫圖 Reward vs. Entropy Coefficient
plt.figure(figsize=(10, 6))

for setting_name in entropy_settings.keys():
    csv_path = f"./logs/ppo_entropy_{setting_name}/monitor.csv"
    if not os.path.exists(csv_path):
        print(f"找不到 log：{csv_path}")
        continue

    df = pd.read_csv(csv_path, skiprows=1)
    if "r" in df.columns:
        smoothed = df["r"].rolling(window=10).mean()
        plt.plot(smoothed, label=f"entropy={setting_name}")
    else:
        print(f"{csv_path} 沒有 reward 資料（r 欄位）")

plt.title("Reward vs. Entropy Coefficient (Smoothed)")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("entropy_vs_reward.png")
plt.show()

print("所有訓練與繪圖完成！")
