import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from evaluate import makeVideo
import os
import numpy as np
import matplotlib.pyplot as plt

# 自訂 Reward Shaping Wrapper
class LunarRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 平穩降落 bonus（角度小、速度慢加分）
        angle_penalty = abs(obs[4])  # 角度
        speed_penalty = abs(obs[2]) + abs(obs[3])  # XY速度
        bonus = max(0, 0.1 - angle_penalty) + max(0, 0.3 - speed_penalty)

        # 減少 crash 懲罰
        if reward == -100:
            reward = -50

        reward += bonus
        return obs, reward, terminated, truncated, info

# 建立 LunarLander 環境並包裝
env = gym.make("LunarLander-v2")
env = LunarRewardWrapper(env)
env = Monitor(env)                      

# # 建立 PPO 模型
# model = PPO("MlpPolicy", env,
#             verbose=1,
#             learning_rate=2.5e-4,
#             n_steps=2048,
#             batch_size=64)

# # 開始訓練
# model.learn(total_timesteps=1_000_000)

# 儲存模型
save_path = "D:/OneDrive - 國立陽明交通大學/桌面/vscode/AICapstone side/ppo_lunarlander_custom"
# model.save(save_path)

# 載入並評估
model = PPO.load(save_path)
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



# 錄影
makeVideo(model, "LunarLander-v2", filename="lunarlander_ppo_custom")
