import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from src.wrappers import DomainRandomizationWrapper
from src.core import logger
from src.config import config


def evaluate_policy(model_path, friction_range=None, mass_range=None):
    env = gym.make(config.ENV_ID)
    
    if friction_range is not None:
        env = DomainRandomizationWrapper(
            env, 
            friction_range=friction_range, 
            mass_scale_range=mass_range,
        )
        
    model = PPO.load(model_path)
    
    episode_rewards = []
    for ep in range(config.NUM_EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
        episode_rewards.append(ep_reward)
        
    return np.mean(episode_rewards), np.std(episode_rewards)

if __name__ == "__main__":
    logger.info("Starting evaluation of trained policies...")
    print("="*60)
    print("TEST EVALUATION MATRIX RESULTS")
    print("="*60)
        
    # 1. Nominal model on Nominal Env
    mean, std = evaluate_policy("models/ppo_nominal", None, None)
    print(f"{'[Nominal Model | Nominal Env]':<35} Reward: {mean:.2f} +/- {std:.2f}")

    # 2. Nominal model on DR Eval
    mean, std = evaluate_policy(
        "models/ppo_nominal",
        friction_range=(config.dr_friction_min, config.dr_friction_max),
        mass_range=(config.dr_mass_min, config.dr_mass_max)
    )
    print(f"{'[Nominal Model | Mild DR Eval]':<35} Reward: {mean:.2f} +/- {std:.2f}")

    # 3. SEVERE DR Eval
    mean, std = evaluate_policy(
        "models/ppo_nominal",
        friction_range=(config.severe_dr_friction_min, config.severe_dr_friction_max),
        mass_range=(config.severe_dr_mass_min, config.severe_dr_mass_max)
    )
    print(f"{'[Nominal Model | SEVERE DR Eval]':<35} Reward: {mean:.2f} +/- {std:.2f}")
    print("-" * 65)

    # 4. DR Model on Nominal 
    mean, std = evaluate_policy("models/ppo_dr", None, None)
    print(f"{'[DR Model | Nominal Env]':<35} Reward: {mean:.2f} +/- {std:.2f}")

    # 5. DR Model on Mild DR
    mean, std = evaluate_policy(
        "models/ppo_dr",
        friction_range=(config.dr_friction_min, config.dr_friction_max),
        mass_range=(config.dr_mass_min, config.dr_mass_max)
    )
    print(f"{'[DR Model | Mild DR Eval]':<35} Reward: {mean:.2f} +/- {std:.2f}")

    # 6. DR Model on SEVERE DR
    mean, std = evaluate_policy(
        "models/ppo_dr",
        friction_range=(config.severe_dr_friction_min, config.severe_dr_friction_max),
        mass_range=(config.severe_dr_mass_min, config.severe_dr_mass_max)
    )
    print(f"{'[DR Model | SEVERE DR Eval]':<35} Reward: {mean:.2f} +/- {std:.2f}")
    print("=" * 65)