import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minihack import MiniHackNavigation, MiniHackRewardManager

# Set up the environment
class CustomMiniHack(MiniHackNavigation):
    def __init__(self):
        super().__init__(
            des_file_path=os.path.abspath("path/to/your/des/file"),
            observation_keys=("glyphs", "chars", "colors", "specials"),
            screen_width=21,
            screen_height=9,
            max_episode_steps=300,
            reward_manager=MiniHackRewardManager()
        )

# Prepare the environment
def make_env():
    def _init():
        env = CustomMiniHack()
        return env

    return _init

# Set up training
env = DummyVecEnv([make_env()])
model = PPO("CnnPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100)

# Save the trained agent
model.save("minihack_agent")

# Load the trained agent for testing
model = PPO.load("minihack_agent")

# Test the agent
env = make_env()()
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
