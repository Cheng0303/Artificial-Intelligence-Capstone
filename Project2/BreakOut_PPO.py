import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from evaluate import evaluate_model, makeVideo, plot_rewards
import ale_py
import gymnasium as gym

time_step = 5_000_000
# make Atari environment
env = gym.make("ALE/Breakout-v5")
env = Monitor(env, f"./logs/{time_step}ppo_breakout.csv")
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)


# # Train PPO model
# model_ppo = PPO("CnnPolicy", env, verbose=1, learning_rate=2.5e-4, n_steps=128, n_epochs=10)
# model_ppo.learn(total_timesteps=time_step)
# model_ppo.save(f"D:/OneDrive - 國立陽明交通大學/桌面/vscode/AICapstone side/{time_step}ppo_breakout")

# Evaluate PPO model
model_ppo = PPO.load(f"D:/OneDrive - 國立陽明交通大學/桌面/vscode/AICapstone side/{time_step}ppo_breakout")
print(type(model_ppo.policy)) 
mean_ppo, std_ppo = evaluate_model(model_ppo, env)
print(f"PPO Avg Reward: {mean_ppo} ± {std_ppo}")

# Make video of PPO model
makeVideo(model_ppo, "ALE/Breakout-v5", f"{time_step}breakout_ppo.gif")

# Plot rewards
log_file = f"./logs/{time_step}ppo_breakout.csv.monitor.csv"
output_file = f"./logs/{time_step}ppo_breakout.png"
plot_rewards(log_file, output_file)

