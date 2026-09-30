import yaml
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class Config:
    env_id: str
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int

    dr_friction_min: float
    dr_friction_max: float
    dr_mass_min: float
    dr_mass_max: float

    severe_friction_min: float
    severe_friction_max: float
    severe_mass_min: float
    severe_mass_max: float

    num_eval_episodes: int
    push_probs: list[float]
    max_push_force: float

    def __post_init__(self):
        # Check if the total_timesteps and learning_rate are positive
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        # Validate the friction and mass ranges for domain randomization 
        if self.dr_friction_min > self.dr_friction_max:
            raise ValueError(
                "dr_friction_min cannot be greater than dr_friction_max."
            )

        if self.dr_mass_min > self.dr_mass_max:
            raise ValueError(
                "dr_mass_min cannot be greater than dr_mass_max."
            )

        # Validate Push Parameters
        if not all(0 <= prob <= 1 for prob in self.push_probs):
            raise ValueError("All push probabilities must be between 0 and 1.")

        if self.max_push_force <= 0:
            raise ValueError("max_push_force must be positive.")


@lru_cache(maxsize=1)
def get_config() -> Config:
    config_path = Path(__file__).parent / "config.yaml"

    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    return Config(**data)

config = get_config()