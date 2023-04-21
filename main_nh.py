import os
import numpy as np
from nle import nethack
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


class NetHackWrapper(nethack.Nethack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space['glyphs']
        self.action_space = self.action_space['command']

    def step(self, action):
        obs, reward, done, info = super().step({'command': action})
        return obs['glyphs'], reward, done, info

    def reset(self):
        obs = super().reset()
        return obs['glyphs']


def main():
    # Create the NetHack environment
    env = NetHackWrapper()

    # Check if the environment is valid
    check_env(env)

    # Create a vectorized environment for Stable Baselines
    vec_env = DummyVecEnv([lambda: env])

    # Set the logging and model directories
    log_dir = 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    model_dir = 'models/'
    os.makedirs(model_dir, exist_ok=True)

    # Initialize the RL model
    model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=log_dir)

    # Create a checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=model_dir)

    # Train the RL model
    model.learn(total_timesteps=int(1e6), callback=checkpoint_callback)

    # Save the trained model
    model.save(os.path.join(model_dir, 'nethack_ppo'))

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')


if __name__ == '__main__':
    main()
