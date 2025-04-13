import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from evaluate import evaluate_model, makeVideo, plot_rewards
import ale_py
import gymnasium as gym
# gym.register_envs(ale_py)

print(gym.__version__)

# make Atari environment
env = gym.make("ALE/Breakout-v5")
env = Monitor(env, "./logs/dqn_breakout.csv")
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

# Train DQN model
model_dqn = DQN("CnnPolicy", env, verbose=1, learning_rate=1e-4, buffer_size=10000, batch_size=32, device="cuda")
model_dqn.learn(total_timesteps=1_000_000)
model_dqn.save("D:/OneDrive - 國立陽明交通大學/桌面/vscode/AICapstone side/dqn_breakout")

# Evaluate DQN model
model_dqn = DQN.load("D:/OneDrive - 國立陽明交通大學/桌面/vscode/AICapstone side/dqn_breakout")
mean_dqn, std_dqn = evaluate_model(model_dqn, env)
print(f"DQN Avg Reward: {mean_dqn} ± {std_dqn}")

# Make video of DQN model
makeVideo(model_dqn, "ALE/Breakout-v5", "breakout_dqn.gif")

# Plot rewards
log_file = "./logs/dqn_breakout.csv.monitor.csv"
output_file = "./logs/dqn_breakout.png"
plot_rewards(log_file, output_file)

