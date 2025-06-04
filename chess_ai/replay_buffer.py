import os
import h5py
import numpy as np

from .config import Config


class ReplayBuffer:
    """HDF5-backed buffer storing (state, policy, value) tuples."""

    def __init__(self, path: str = Config.REPLAY_BUFFER_PATH, max_size: int = Config.REPLAY_BUFFER_SIZE):
        self.path = path
        self.max_size = max_size
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.file = h5py.File(self.path, "a")
        if "states" not in self.file:
            self.file.create_dataset(
                "states",
                (self.max_size, Config.NUM_CHANNELS, 8, 8),
                dtype="float32",
            )
            self.file.create_dataset("pis", (self.max_size, 4672), dtype="float32")
            self.file.create_dataset("zs", (self.max_size,), dtype="float32")
        self.current_size = self.file["zs"].attrs.get("current_size", 0)

    def add(self, state: np.ndarray, pi: np.ndarray, z: float) -> None:
        idx = self.current_size % self.max_size
        self.file["states"][idx] = state
        self.file["pis"][idx] = pi
        self.file["zs"][idx] = z
        self.current_size += 1
        self.file["zs"].attrs["current_size"] = self.current_size
        self.file.flush()

    def sample(self, batch_size: int):
        max_idx = min(self.current_size, self.max_size)
        indices = np.random.choice(max_idx, batch_size, replace=False)
        states = self.file["states"][indices]
        pis = self.file["pis"][indices]
        zs = self.file["zs"][indices]
        return states, pis, zs

    def __len__(self) -> int:
        return min(self.current_size, self.max_size)
