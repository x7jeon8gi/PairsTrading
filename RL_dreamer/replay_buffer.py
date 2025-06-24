import numpy as np
from typing import Dict, Tuple

class ReplayBuffer:
    """
    Replay buffer for storing and sampling sequences of experiences for the world model.
    """
    def __init__(self, config: Dict, observation_dim: int, action_dim: int):
        """
        Initializes the replay buffer.

        Args:
            config (Dict): Configuration dictionary containing buffer settings.
            observation_dim (int): The dimension of the observation space.
            action_dim (int): The dimension of the action space.
        """
        self.capacity = int(config['capacity'])
        self.batch_size = int(config['batch_size'])
        self.sequence_length = int(config['sequence_length'])
        
        self.observations = np.empty((self.capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.terminals = np.empty(self.capacity, dtype=np.bool_)
        
        self.current_size = 0
        self.pointer = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        """
        Adds a single transition to the buffer.
        
        Args:
            obs (np.ndarray): The observation from the environment.
            action (np.ndarray): The action taken by the agent.
            reward (float): The reward received.
            done (bool): Whether the episode has terminated.
        """
        self.observations[self.pointer] = obs
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.terminals[self.pointer] = done
        
        self.pointer = (self.pointer + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self) -> Dict[str, np.ndarray]:
        """
        Samples a batch of sequences from the buffer.

        Returns:
            A dictionary containing batches of sequences for observations, actions,
            rewards, and terminals.
        """
        sequence_indices = []
        while len(sequence_indices) < self.batch_size:
            # -1 to ensure there's a reward and next_state for the final step of the sequence
            start_idx = np.random.randint(0, self.current_size - self.sequence_length)
            
            # The sequence should not wrap around the buffer's pointer
            if start_idx + self.sequence_length > self.pointer and start_idx < self.pointer:
                 continue

            # The sequence should not cross an episode boundary
            if np.any(self.terminals[start_idx:start_idx + self.sequence_length -1]):
                continue

            sequence_indices.append(start_idx)
        
        # Collect sequences
        batch_obs, batch_actions, batch_rewards, batch_terminals = [], [], [], []
        for start_idx in sequence_indices:
            indices = np.arange(start_idx, start_idx + self.sequence_length)
            batch_obs.append(self.observations[indices])
            batch_actions.append(self.actions[indices])
            batch_rewards.append(self.rewards[indices])
            batch_terminals.append(self.terminals[indices])

        return {
            "observations": np.array(batch_obs, dtype=np.float32),
            "actions": np.array(batch_actions, dtype=np.float32),
            "rewards": np.array(batch_rewards, dtype=np.float32),
            "terminals": np.array(batch_terminals, dtype=np.bool_),
        }

    def __len__(self) -> int:
        return self.current_size
