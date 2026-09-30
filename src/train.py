import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.wrappers import DomainRandomizationWrapper
from src.core import logger
from src.config import config


def train_policy(use_dr):
    env = gym.make(config.env_id)
    
    model_name = "ppo_dr" if use_dr else "ppo_nominal"
    if use_dr:
        env = DomainRandomizationWrapper(
            env, 
            friction_range=(config.dr_friction_min, config.dr_friction_max), 
            mass_scale_range=(config.dr_mass_min, config.dr_mass_max)
        )

    model = PPO(
        "MlpPolicy", 
        env, 
        device="cpu",
        verbose=0,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps, 
        batch_size=config.batch_size,
        tensorboard_log=f"./logs/{model_name}",
        seed=config.seed,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=f"./models/checkpoints/{model_name}",
        name_prefix=model_name,
    )
    logger.info(f"Training {model_name} for {config.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        progress_bar=True,
        callback=checkpoint_callback,
    )
    model.save(f"models/{model_name}")
    logger.info(f"Saved {model_name} model to `models/{model_name}`")

if __name__ == "__main__":
    # Train both nominal and domain-randomized policies
    train_policy(use_dr=False)
    train_policy(use_dr=True)

    