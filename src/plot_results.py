from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.core import logger


def visualize_rewards(training_mode: Literal["ppo_dr", "ppo_nominal"],):
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
            f"No log directory found: {log_path}"
        )
    
    event_files = list(log_path.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found in: {log_path}"
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

    # Save to assets/
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    output_path = assets_dir / f"training_progress_{training_mode}.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    logger.info(f"Saved the episode reward and length plot to: {output_path}")

    plt.show()
    plt.close(fig)



def compare_environments():
    """
    Compare the performance of the nominal and DR-trained policies across different environments.
    """
    logger.info("Comparing nominal and DR-trained policies across environments...")
    pass


if __name__ == "__main__":
    visualize_rewards("ppo_nominal")
    visualize_rewards("ppo_dr")
    compare_environments()