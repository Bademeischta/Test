import os
import tempfile

import numpy as np

from chess_ai.lmdb_replay_buffer import LMDBReplayBuffer


def test_lmdb_replay_buffer_add_and_sample(tmp_path):
    path = tmp_path / "buffer.lmdb"
    buf = LMDBReplayBuffer(str(path), capacity=10)
    for i in range(5):
        buf.add(i, i + 0.1, i + 0.2)

    states, policies, values = buf.sample(3)
    assert len(states) == 3
    assert len(buf) == 5


def test_lmdb_replay_buffer_prioritized(tmp_path):
    path = tmp_path / "buffer.lmdb"
    buf = LMDBReplayBuffer(str(path), capacity=5)
    for i in range(5):
        buf.add(i, i + 0.1, i + 0.2, priority=float(i + 1))

    np.random.seed(0)
    states1, _, _ = buf.sample_prioritized(3)
    np.random.seed(0)
    states2, _, _ = buf.sample_prioritized(3)
    assert states1 == states2
