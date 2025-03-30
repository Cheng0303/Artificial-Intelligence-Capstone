# Train DQN model
# model_dqn = DQN("CnnPolicy", env, verbose=1, learning_rate=1e-4, buffer_size=10000)
# model_dqn.learn(total_timesteps=10000)
# model_dqn.save("dqn_breakout")

# # Evaluate DQN model
# model_dqn = DQN.load("dqn_breakout")
# mean_dqn, std_dqn = evaluate_model(model_dqn, env)
# print(f"DQN Avg Reward: {mean_dqn} ± {std_dqn}")

# # Make video of DQN model
# makeVideo(model_dqn, env, "breakout_dqn.gif")