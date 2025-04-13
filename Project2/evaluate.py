import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3 import DQN, PPO
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import os
import pandas as pd
import matplotlib.pyplot as plt



def evaluate_model(model, env, num_episodes=10):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset() 
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        
        episode_rewards.append(total_reward)
    
    return np.mean(episode_rewards), np.std(episode_rewards)


def makeVideo(model, envname, filename="breakout_dqn.gif"):

    video_folder = "./videos/"
    os.makedirs(video_folder, exist_ok=True)


    if envname == "ALE/Breakout-v5":

        test_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        # test_env = RecordVideo(test_env, video_folder=video_folder, episode_trigger=lambda e: True)  
        test_env = DummyVecEnv([lambda: test_env])
        test_env = VecFrameStack(test_env, n_stack=4)

    else:

        test_env = gym.make(envname, render_mode="rgb_array")
        test_env = DummyVecEnv([lambda: test_env])



    test_env = VecVideoRecorder(
        test_env, 
        video_folder=video_folder, 
        record_video_trigger=lambda x: x == 0, 
        video_length=500,
        name_prefix=filename
    )


    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)

    test_env.close()
    print("測試並錄影完成，請查看 videos 資料夾。")

def plot_rewards(log_file, output_file):
    # Read the CSV file
    df = pd.read_csv(log_file, skiprows=1)

    # Plot the rewards
    plt.figure(figsize=(10,5))
    plt.plot(df["r"], label="Episode Reward", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Reward over Time")
    plt.legend()
    plt.savefig(output_file)
    plt.show()
