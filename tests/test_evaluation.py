import torch

from chess_ai.evaluation import evaluate
from chess_ai.action_index import ACTION_SIZE
from chess_ai.config import Config


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch = x.size(0)
        log_p = torch.log_softmax(torch.ones(batch, ACTION_SIZE), dim=1)
        v = torch.zeros(batch, 1)
        return log_p, v


def test_evaluate_runs():
    net = DummyNet()
    original_epsilon = Config.DIRICHLET_EPSILON
    Config.DIRICHLET_EPSILON = 0.0
    try:
        stats = evaluate(net, net, num_games=1, num_simulations=1)
    finally:
        Config.DIRICHLET_EPSILON = original_epsilon
    assert set(stats.keys()) == {"wins", "losses", "draws"}
    assert sum(stats.values()) == 1
