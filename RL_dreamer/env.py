import gym
from gym import spaces
import numpy as np
import os
import sys
from typing import Dict

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.env import TradingEnvironment

class DreamerEnv:
    """
    A wrapper for the TradingEnvironment to make it compatible with the Dreamer agent.
    """
    def __init__(self, env_config: Dict, dreamer_config: Dict):
        """
        Initializes the wrapped environment.

        Args:
            env_config (Dict): Configuration for the underlying TradingEnvironment.
            dreamer_config (Dict): The full configuration for the Dreamer agent.
        """
        self.env = TradingEnvironment(**env_config)
        self._config = dreamer_config
        
        # Define observation and action spaces based on the config
        # obs_dim is now 'num_inputs' in env_config
        obs_dim = env_config['num_inputs']
        # action_dim is now at the root of the dreamer_config
        action_dim = self._config['action_dim']
        
        # Assuming the action space from SAC config: [[min1, min2], [max1, max2]]
        # We can extract these from the old SAC config if needed, or define here.
        # For now, using the values from RL/SAC_config.yaml
        sac_action_space_raw = [[0.0, 0.0] , [2.0, 0.75]] 
        action_low = np.array(sac_action_space_raw[0], dtype=np.float32)
        action_high = np.array(sac_action_space_raw[1], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """
        Resets the environment and returns the initial observation.
        """
        obs = self.env.reset()
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Takes a step in the environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            A tuple containing (observation, reward, done, info).
            The 'info' dictionary is empty for now.
        """
        # The underlying env might not handle np.float64 actions well
        action = action.astype(np.float32)
        
        # The original env returns (next_state, reward, done)
        next_obs, reward, done = self.env.step(action)
        
        return next_obs.astype(np.float32), reward, done, {}

    def render(self, mode='human'):
        """
        Rendering is not supported.
        """
        pass

    def close(self):
        """
        Closes the environment.
        """
        pass
