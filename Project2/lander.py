import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from evaluate import makeVideo
import matplotlib.pyplot as plt
import numpy as np

# 建立 LunarLander 環境
env = gym.make("LunarLander-v2")
env = Monitor(env)  # 加入 Monitor 來記錄 reward 等資訊

# 建立 PPO 模型
model = PPO("MlpPolicy", env, verbose=1, learning_rate=2.5e-4, n_steps=2048, batch_size=64)

# 開始訓練
model.learn(total_timesteps=1_000_000)

# 儲存模型
model.save("D:/OneDrive - 國立陽明交通大學/桌面/vscode/AICapstone side/ppo_lunarlander")

model = PPO.load("D:/OneDrive - 國立陽明交通大學/桌面/vscode/AICapstone side/ppo_lunarlander")
rewards, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, return_episode_rewards=True)
print(f"Avg Reward：{np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


plt.figure(figsize=(8, 5))
plt.plot(rewards, marker='o')
plt.title("Evaluation Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_lunarlander_eval_rewards.png")
plt.show()

makeVideo(model, "LunarLander-v2", "lunarlander_ppo.gif")

