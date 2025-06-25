import random
import numpy as np
import os
import pickle

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    # def save_buffer(self, env_name, suffix="", save_path=None):
    #     if not os.path.exists('checkpoints/'):
    #         os.makedirs('checkpoints/')

    #     if save_path is None:
    #         save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
    #     print('Saving buffer to {}'.format(save_path))

    #     with open(save_path, 'wb') as f:
    #         pickle.dump(self.buffer, f)

    # def load_buffer(self, save_path):
    #     print('Loading buffer from {}'.format(save_path))

    #     with open(save_path, "rb") as f:
    #         self.buffer = pickle.load(f)
    #         self.position = len(self.buffer) % self.capacity

# --- Prioritized Experience Replay ---

class SumTree:
    """
    SumTree data structure for efficient sampling based on priorities.
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayMemory:
    """
    Replay memory that uses priorities for sampling (PER).
    """
    e = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] converts the TD error to priority
    beta = 0.4  # importance-sampling, from initial value to 1
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, seed, alpha=0.6, beta=0.4):
        random.seed(seed)
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the memory with max priority.
        """
        data = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, data)

    def sample(self, batch_size):
        """
        Samples a batch of experiences based on their priorities.
        Returns the batch, their indices in the tree, and IS weights.
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            if isinstance(data, int) and data == 0: # Workaround for initial zeroed data
                # Resample if we hit an empty spot (can happen at the beginning)
                s = random.uniform(0, self.tree.total())
                (idx, p, data) = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (state, action, reward, next_state, done), idxs, is_weight
    
    def update_priorities(self, tree_idxs, td_errors):
        """
        Updates the priorities of the sampled experiences.
        """
        priorities = (np.abs(td_errors) + self.e) ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))

        for i, idx in enumerate(tree_idxs):
            self.tree.update(idx, priorities[i])

    def __len__(self):
        return self.tree.n_entries