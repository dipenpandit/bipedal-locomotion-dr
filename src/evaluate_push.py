import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from wrappers import PushRecoveryWrapper
from src.core import logger
from src.config import config


def evaluate_push_recovery(model_path, num_episodes=100):
    env = gym.make("2d-v5")
    # Apply a 5% chance per step of a push up to 50 Newtons
    env = PushRecoveryWrapper(env, push_prob=config.push_prob, max_force=config.max_push_force)
        
    model = PPO.load(model_path)
    
    survival_lengths = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_length += 1
            done = terminated or truncated
            
        survival_lengths.append(ep_length)
        
    mean_length = np.mean(survival_lengths)
    std_length = np.std(survival_lengths)
    
    print(f"[{model_path}]")
    print(f"  Mean Survival Steps: {mean_length:.1f} +/- {std_length:.1f}")
    return mean_length, std_length

if __name__ == "__main__":
    print("="*50)
    print("RUNNING PUSH RECOVERY EVALUATION")
    print("="*50)
    
    evaluate_push_recovery("models/ppo_nominal")
    print("-" * 50)
    evaluate_push_recovery("models/ppo_dr")
    print("="*50)