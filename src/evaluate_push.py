import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import json
from pathlib import Path
from collections import defaultdict
from src.wrappers import PushRecoveryWrapper
from src.config import config
from src.core import logger

def evaluate_push(model_path, push_probs):
    """Uses different push probabilities and returns mean survival steps for each."""
    model = PPO.load(model_path, device="cpu")
    
    results = {}
    
    for prob in push_probs:
        # We recreate the env for each probability to ensure clean state
        env = gym.make(config.env_id)
        env = PushRecoveryWrapper(env, push_prob=prob, max_force=config.max_push_force)
        
        survival_lengths = []
        for ep in range(config.num_eval_episodes):
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
        results[prob] = (mean_length, std_length)
        print(f"Push Prob {prob:.2f}: {mean_length:.1f} +/- {std_length:.1f} steps")
        
    return results


def run_and_save_push_evaluation():
    print("="*65)
    print("RUNNING PUSH PROBABILITY ABLATION STUDY")
    print("="*65)

    all_results = {}
    
    print("[Nominal Model]")
    all_results["nominal_push"] = evaluate_push("models/ppo_nominal", config.push_probs)
    
    print("\n[DR Model]")
    all_results["dr_push"] = evaluate_push("models/ppo_dr", config.push_probs)    
    print("\n" + "="*65)

    # Save the results to JSON
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    results_file = assets_dir / "push_sweep.json"

    with results_file.open("w") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Push sweep results saved to {results_file}")


if __name__ == "__main__":
    run_and_save_push_evaluation()