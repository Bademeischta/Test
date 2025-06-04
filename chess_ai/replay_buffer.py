import random
from collections import deque

from .config import Config


class ReplayBuffer:
    def __init__(self, capacity: int = Config.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)
