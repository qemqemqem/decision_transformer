import random
import time

import gym
import nle
env = gym.make("NetHackScore-v0")
env.reset()  # each reset generates a new dungeon
while True:
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        break
    # env.step(1)  # move agent '@' north
    env.render()
    time.sleep(0.1)