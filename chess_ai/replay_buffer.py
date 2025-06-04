import random
from collections import deque
from typing import Sequence

import numpy as np

from .config import Config


class ReplayBuffer:
    def __init__(self, capacity: int = Config.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        """Return a batch of samples from the buffer.

        Numpy indices are sorted before indexing so that usage with HDF5
        datasets works without raising ``TypeError``.
        """

        if batch_size > len(self.buffer):
            raise ValueError("Batch size larger than buffer")

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        indices = np.sort(indices)
        batch = [self.buffer[i] for i in indices]
        states, policies, values = zip(*batch)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)
