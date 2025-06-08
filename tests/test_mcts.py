import chess
import torch

from chess_ai.action_index import ACTION_SIZE, index_to_move, move_to_index
from chess_ai.mcts import MCTS


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch = x.size(0)
        log_p = torch.log_softmax(torch.ones(batch, ACTION_SIZE), dim=1)
        v = torch.zeros(batch, 1)
        return log_p, v


def test_mcts_returns_valid_move_indices():
    board = chess.Board()
    net = DummyNet()
    mcts = MCTS(net, num_simulations=2)
    visit_counts = mcts.run(board)

    legal_indices = {move_to_index(m) for m in board.legal_moves}
    assert set(visit_counts.keys()) == legal_indices
    for idx in visit_counts:
        move = index_to_move(idx)
        assert move in board.legal_moves
