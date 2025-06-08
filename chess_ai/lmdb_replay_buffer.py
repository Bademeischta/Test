import os
import pickle

import lmdb
import numpy as np

from .config import Config


class LMDBReplayBuffer:
    """Replay buffer backed by LMDB for large datasets."""

    META_NEXT = b"next"
    META_SIZE = b"size"

    def __init__(
        self,
        path: str,
        capacity: int = Config.REPLAY_BUFFER_SIZE,
        map_size: int = 1 << 30,
    ):
        self.env = lmdb.open(os.path.expanduser(path), map_size=map_size)
        self.capacity = capacity
        with self.env.begin(write=True) as txn:
            if txn.get(self.META_NEXT) is None:
                txn.put(self.META_NEXT, b"0")
                txn.put(self.META_SIZE, b"0")

    def _meta(self, key: bytes) -> int:
        with self.env.begin() as txn:
            val = txn.get(key)
            return int(val.decode()) if val is not None else 0

    def _set_meta(self, key: bytes, value: int):
        with self.env.begin(write=True) as txn:
            txn.put(key, str(value).encode())

    def add(self, state, policy, value, priority: float | None = None):
        if priority is None:
            priority = abs(value) + 1e-5
        next_idx = self._meta(self.META_NEXT)
        size = self._meta(self.META_SIZE)
        with self.env.begin(write=True) as txn:
            txn.put(f"d{next_idx}".encode(), pickle.dumps((state, policy, value)))
            txn.put(f"p{next_idx}".encode(), pickle.dumps(priority))
            next_idx = (next_idx + 1) % self.capacity
            size = min(size + 1, self.capacity)
            txn.put(self.META_NEXT, str(next_idx).encode())
            txn.put(self.META_SIZE, str(size).encode())

    def __len__(self) -> int:
        return self._meta(self.META_SIZE)

    def _get_priorities(self, size):
        priorities = np.empty(size, dtype=np.float32)
        with self.env.begin() as txn:
            for i in range(size):
                data = txn.get(f"p{i}".encode())
                if data is None:
                    priorities[i] = 1e-5
                else:
                    priorities[i] = pickle.loads(data)
        return priorities

    def _load_batch(self, indices):
        with self.env.begin() as txn:
            batch = [pickle.loads(txn.get(f"d{i}".encode())) for i in indices]
        states, policies, values = zip(*batch)
        return states, policies, values

    def sample(self, batch_size):
        size = len(self)
        if batch_size > size:
            raise ValueError("Batch size larger than buffer")
        indices = np.random.choice(size, size=batch_size, replace=False)
        indices = np.sort(indices)
        return self._load_batch(indices)

    def sample_prioritized(self, batch_size, alpha: float = 0.6):
        size = len(self)
        if batch_size > size:
            raise ValueError("Batch size larger than buffer")
        priorities = self._get_priorities(size)
        probs = priorities**alpha
        probs /= probs.sum()
        indices = np.random.choice(size, size=batch_size, p=probs)
        indices = np.sort(indices)
        return self._load_batch(indices)
