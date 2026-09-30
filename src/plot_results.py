import json
from typing import Literal
import numpy as np
from pathlib import Path
import matplotlib 
matplotlib.use("Agg")  # Use a non-interactive backend for environments without a display
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.core import logger
from src.config import config



# Load results from JSON files
def load_results(json_file: Literal["eval_matrix.json", "push_sweep.json"]):
    try:
        with open(f"assets/{json_file}", "r") as f:
                return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {json_file}. Make sure to run the evaluation scripts first.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {json_file}. Make sure the file is valid JSON.")
        raise



# Visualize training progress from tensorboard logs
def plot_training_progress(training_mode: Literal["ppo_dr", "ppo_nominal"],):
    """ 
    Visualize the episode reward and length from tensorboard logs.
    """
    logger.info("Comparing Nominal and DR training episode reward and length...")
    # Validate training_mode
    if training_mode not in ["ppo_nominal", "ppo_dr"]:
        raise ValueError(
            f"Invalid training_mode: '{training_mode}'. "
            "Must be 'ppo_nominal' or 'ppo_dr'."
        )

    # Find the latest tensorboard event file in the logs directory
    log_path = Path("logs") / training_mode

    if not log_path.exists():
        raise FileNotFoundError(
            f"No log directory found: {log_path}. Make sure to run the training scripts first."
        )
    
    event_files = list(log_path.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found in: {log_path}. Make sure to run the training scripts first."
        )

    latest_event_file = max(
        event_files, 
        key=lambda p: p.stat().st_mtime
    )

    event_acc = EventAccumulator(str(latest_event_file))
    event_acc.Reload()

    # Fetch the metrics for episode reward mean and episode length mean
    def get_metric(tag):
        events = event_acc.Scalars(tag)
        return [e.step for e in events], [e.value for e in events]

    steps_reward, rewards = get_metric("rollout/ep_rew_mean")
    steps_len, lengths = get_metric("rollout/ep_len_mean")

    # Plotting
    title = (
        "Domain Randomization Training"
        if training_mode == "ppo_dr"
        else "Nominal Training"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{title} Progress", fontsize=14)

    axes[0].plot(steps_reward, rewards, color="blue")
    axes[0].set_title("Episode Reward Mean")
    axes[0].set_xlabel("Timesteps")
    axes[0].grid(True)

    axes[1].plot(steps_len, lengths, color="green")
    axes[1].set_title("Episode Length Mean")
    axes[1].set_xlabel("Timesteps")
    axes[1].grid(True)

    plt.tight_layout()

    # Save the figure to the assets directory
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    output_path = assets_dir / f"fig1_training_progress_{training_mode}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved the episode reward and length plot to: {output_path}")

    plt.show()
    plt.close(fig)



# Visualize the evaluation results across different environments
def plot_eval_matrix(results: dict):
    """
    Compare the performance of the nominal and DR-trained policies across different environments.
    """
    logger.info("Comparing nominal and DR-trained policies across environments...")
        
    labels = ['Nominal Env', 'Mild DR', 'Severe DR']
    x = np.arange(len(labels))
    width = 0.35
    
    nom_means = [
        results["nominal"]["nominal_env"]["mean"],
        results["nominal"]["mild_dr"]["mean"],
        results["nominal"]["severe_dr"]["mean"]
    ]
    nom_stds = [
        results["nominal"]["nominal_env"]["std"],
        results["nominal"]["mild_dr"]["std"],
        results["nominal"]["severe_dr"]["std"]
    ]
    dr_means = [
        results["dr"]["nominal_env"]["mean"],
        results["dr"]["mild_dr"]["mean"],
        results["dr"]["severe_dr"]["mean"]
    ]
    dr_stds = [
        results["dr"]["nominal_env"]["std"],
        results["dr"]["mild_dr"]["std"],
        results["dr"]["severe_dr"]["std"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, nom_means, width, label='Nominal Train', color='skyblue', yerr=nom_stds, capsize=5)
    rects2 = ax.bar(x + width/2, dr_means, width, label='DR Train', color='lightcoral', yerr=dr_stds, capsize=5)
    
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Robustness Evaluation Matrix', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig("assets/fig2_robustness_matrix.png", dpi=600)

    plt.show()
    plt.close()



# Visualize the push probability ablation study results
def plot_push_sweep(results: dict):
    """
    Visualize the results of the push probability ablation study.
    """
    logger.info("Visualizing push probability ablation study results...")
    probs = config.push_probs
    nom_means = [results["nominal_push"][str(p)]["mean"] for p in probs]
    nom_stds = [results["nominal_push"][str(p)]["std"] for p in probs]
        
    dr_means = [results["dr_push"][str(p)]["mean"] for p in probs]
    dr_stds = [results["dr_push"][str(p)]["std"] for p in probs]
        
    fig, ax = plt.subplots(figsize=(10, 6))
        
    ax.errorbar(probs, nom_means, yerr=nom_stds, label='Nominal Train', color='skyblue', marker='o', capsize=5, linewidth=2)
    ax.errorbar(probs, dr_means, yerr=dr_stds, label='DR Train', color='lightcoral', marker='s', capsize=5, linewidth=2)
        
    ax.set_xlabel('Push Probability per Step', fontsize=12)
    ax.set_ylabel('Mean Survival Steps', fontsize=12)
    ax.set_title('Push Recovery Ablation Study', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    fig.savefig("assets/fig3_push_ablation.png", dpi=600)

    plt.show()
    plt.close()



if __name__ == "__main__":
    plot_training_progress("ppo_nominal")
    plot_training_progress("ppo_dr")
    matrix = load_results("eval_matrix.json")
    push = load_results("push_sweep.json")
    plot_eval_matrix(matrix)
    plot_push_sweep(push)