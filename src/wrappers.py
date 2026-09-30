import gymnasium as gym
import mujoco
import numpy as np
from src.core import logger
from src.config import config


class PhysicsWrapper(gym.Wrapper):
    """Allows manual adjustment of physical properties (friction, mass)."""

    FOOT_GEOMS = ["foot_geom", "foot_left_geom"]

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.model = env.unwrapped.model
        m = self.model

        # Fetch IDs using names
        self.floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in self.FOOT_GEOMS
        ]
        self.torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # Check if the names were correct (-1 id means naming error)
        assert (
            self.floor_id >= 0 and self.torso_id >= 0 and min(self.foot_ids) >= 0
        ), "Name not found - check XML names."

        # Save original values to prevent drift
        self.nominal_friction = m.geom_friction[:, 0].copy()
        self.nominal_mass = m.body_mass[self.torso_id]
        self.nominal_inertia = m.body_inertia[self.torso_id].copy()
        logger.debug("PhysicsWrapper initialized.")

    def set_friction(self, value):
        """
        Set effective sliding friction between feet and floor.

        MuJoCo uses the MAX of the friction values of the two
        colliding geoms. To change effective sliding friction, we must update
        both the floor and the robot's feet.
        """
        self.model.geom_friction[self.floor_id, 0] = value
        for foot_gid in self.foot_ids:
            self.model.geom_friction[foot_gid, 0] = value

    def set_torso_mass_scale(self, scale):
        """Multiply torso mass and inertia by `scale`."""
        self.model.body_mass[self.torso_id] = self.nominal_mass * scale
        self.model.body_inertia[self.torso_id] = self.nominal_inertia * scale

    def restore_nominal(self):
        """Reset to original XML values."""
        self.model.geom_friction[:, 0] = self.nominal_friction
        self.set_torso_mass_scale(1.0)
        logger.info("Restored nominal physics parameters.")


class DomainRandomizationWrapper(PhysicsWrapper):
    """Samples new friction + torso mass at the start of EVERY episode."""
    def __init__(
            self,
            env:gym.Env,
            friction_range=(config.dr_friction_min, config.dr_friction_max),
            mass_scale_range=(config.dr_mass_min, config.dr_mass_max),
        ):
        super().__init__(env)
        self.friction_range = friction_range
        self.mass_scale_range = mass_scale_range
        self.current_params = {}
        self.rng = np.random.default_rng()

    def reset(self, **kwargs):
        # Handle seeding if passed
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Sample new physics
        friction = self.rng.uniform(*self.friction_range)
        mass_scale = self.rng.uniform(*self.mass_scale_range)

        # Apply to simulator
        self.set_friction(friction)
        self.set_torso_mass_scale(mass_scale)
        self.current_params = {"friction": friction, "mass_scale": mass_scale}

        # Normal reset
        obs, info = super().reset(**kwargs) # invokes .reset() of parent class
        info["dr_params"] = self.current_params
        return obs, info


class PushRecoveryWrapper(gym.Wrapper):
    """
    Applies a random horizontal push to the robot's torso at random intervals 
    to test the policy's ability to recover from external disturbances.
    """
    def __init__(self, env, push_prob, max_force=config.max_push_force):
        super().__init__(env)
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
            
        # Get the torso body ID
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        assert (
            self.torso_id >= 0
        ),"Torso body not found."
            
        self.push_prob = push_prob
        self.max_force = max_force
        self.rng = np.random.default_rng()

    def step(self, action):
        # 1. Decide whether to push
        if self.rng.random() < self.push_prob:
            # Apply random horizontal force (positive = forward push, negative = backward push)
            force = self.rng.uniform(-self.max_force, self.max_force)
            self.data.xfrc_applied[self.torso_id, 0] = force
            logger.debug(f"Push applied: force={force:.2f}")  # debug level to avoid spam
        else:
            # Clear any previous force
            self.data.xfrc_applied[self.torso_id, 0] = 0.0
                
        # 2. Execute the environment step
        obs, reward, terminated, truncated, info = self.env.step(action)
            
        # Add push info to the info dict for logging
        info["pushed"] = self.data.xfrc_applied[self.torso_id, 0] != 0.0
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Ensure no lingering forces on reset
        self.data.xfrc_applied[self.torso_id, 0] = 0.0
        obs, info = super().reset(**kwargs)     # invokes .reset() of parent class
        return obs, info