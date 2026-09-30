import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path
import json
from collections import defaultdict
from src.wrappers import DomainRandomizationWrapper
from src.core import logger
from src.config import config


def evaluate_policy(model_path, friction_range=None, mass_range=None):
    env = gym.make(config.env_id)
    
    if friction_range is not None:
        env = DomainRandomizationWrapper(
            env, 
            friction_range=friction_range, 
            mass_scale_range=mass_range,
        )
        
    model = PPO.load(
        model_path, 
        device="cpu",
    )
    
    episode_rewards = []
    for ep in range(config.num_eval_episodes):
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


def run_and_save_evaluation():
    logger.info("Starting evaluation of trained policies...")
    results = defaultdict(dict)

    print("="*65)
    print("TEST EVALUATION MATRIX RESULTS")
    print("="*65)
        
    # 1. Nominal model on Nominal Env
    mean, std = evaluate_policy("models/ppo_nominal", None, None)
    print(f"{'[Nominal Model | Nominal Env]':<35} Reward: {mean:.2f} +/- {std:.2f}")
    results["nominal"]["nominal_env"] = {"mean": mean, "std": std}

    # 2. Nominal model on DR Eval
    mean, std = evaluate_policy(
        "models/ppo_nominal",
        friction_range=(config.dr_friction_min, config.dr_friction_max),
        mass_range=(config.dr_mass_min, config.dr_mass_max)
    )
    print(f"{'[Nominal Model | Mild DR Eval]':<35} Reward: {mean:.2f} +/- {std:.2f}")
    results["nominal"]["mild_dr"] = {"mean": mean, "std": std}

    # 3. Nominal model on SEVERE DR Env
    mean, std = evaluate_policy(
        "models/ppo_nominal",
        friction_range=(config.severe_friction_min , config.severe_friction_max),
        mass_range=(config.severe_mass_min, config.severe_mass_max)
    )
    print(f"{'[Nominal Model | SEVERE DR Eval]':<35} Reward: {mean:.2f} +/- {std:.2f}")
    results["nominal"]["severe_dr"] = {"mean": mean, "std": std}

    print("-" * 65)

    # 4. DR Model on Nominal Env
    mean, std = evaluate_policy("models/ppo_dr", None, None)
    print(f"{'[DR Model | Nominal Env]':<35} Reward: {mean:.2f} +/- {std:.2f}")
    results["dr"]["nominal_env"] = {"mean": mean, "std": std}


    # 5. DR Model on Mild DR Env
    mean, std = evaluate_policy(
        "models/ppo_dr",
        friction_range=(config.dr_friction_min, config.dr_friction_max),
        mass_range=(config.dr_mass_min, config.dr_mass_max)
    )
    print(f"{'[DR Model | Mild DR Eval]':<35} Reward: {mean:.2f} +/- {std:.2f}")
    results["dr"]["mild_dr"] = {"mean": mean, "std": std}

    # 6. DR Model on SEVERE DR Env
    mean, std = evaluate_policy(
        "models/ppo_dr",
        friction_range=(config.severe_friction_min , config.severe_friction_max),
        mass_range=(config.severe_mass_min, config.severe_mass_max)
    )
    print(f"{'[DR Model | SEVERE DR Eval]':<35} Reward: {mean:.2f} +/- {std:.2f}")
    results["dr"]["severe_dr"] = {"mean": mean, "std": std}

    print("=" * 65)

    # Save the results to JSON
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    results_file = assets_dir / "eval_matrix.json"

    with results_file.open("w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    run_and_save_evaluation()