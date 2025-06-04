import numpy as np
from chess_ai.replay_buffer import ReplayBuffer
from chess_ai.config import Config


def test_replay_buffer_add_and_sample(tmp_path):
    path = tmp_path / "buffer.h5"
    buffer = ReplayBuffer(path=str(path), max_size=10)
    state = np.zeros((Config.NUM_CHANNELS, 8, 8), dtype=np.float32)
    pi = np.zeros(4672, dtype=np.float32)
    for i in range(5):
        buffer.add(state, pi, float(i))
    states, pis, zs = buffer.sample(3)
    assert states.shape[1:] == (Config.NUM_CHANNELS, 8, 8)
    assert pis.shape[1] == 4672
    assert len(zs) == 3
