import numpy as np

from chess_ai.replay_buffer import ReplayBuffer


def test_sample_sorts_indices():
    np.random.seed(0)
    buffer = ReplayBuffer(capacity=10)
    for i in range(10):
        buffer.add(i, i + 0.1, i + 0.2)

    np.random.seed(0)
    states, policies, values = buffer.sample(5)

    np.random.seed(0)
    expected_indices = np.random.choice(10, size=5, replace=False)
    expected_indices = np.sort(expected_indices)

    assert states == tuple(expected_indices)
    assert policies == tuple(i + 0.1 for i in expected_indices)
    assert values == tuple(i + 0.2 for i in expected_indices)
