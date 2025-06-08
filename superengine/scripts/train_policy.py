import glob
import os

import chess
import chess.pgn
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from chess_ai.action_index import ACTION_SIZE, move_to_index
from chess_ai.game_environment import GameEnvironment
from chess_ai.policy_value_net import PolicyValueNet


def load_games(game_dir: str):
    """Load PGN files from ``game_dir`` and return tensors for training."""
    states, policies, values = [], [], []
    for pgn_file in glob.glob(os.path.join(game_dir, "*.pgn")):
        with open(pgn_file) as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                result = game.headers.get("Result", "1/2-1/2")
                if result == "1-0":
                    winner = 1
                elif result == "0-1":
                    winner = -1
                else:
                    winner = 0
                board = game.board()
                for move in game.mainline_moves():
                    state = GameEnvironment.encode_board(board)
                    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
                    policy[move_to_index(move)] = 1.0
                    value = winner if board.turn == chess.WHITE else -winner
                    states.append(state)
                    policies.append(policy)
                    values.append(value)
                    board.push(move)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    policies = torch.tensor(np.array(policies), dtype=torch.float32)
    values = torch.tensor(np.array(values), dtype=torch.float32)
    return TensorDataset(states, policies, values)


class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PolicyValueNet(GameEnvironment.NUM_CHANNELS, ACTION_SIZE)
        self.loss_mse = torch.nn.MSELoss()

    def training_step(self, batch, _):
        s, p_target, v_target = batch
        log_p, v = self.net(s)
        loss_p = -(p_target * log_p).sum(dim=1).mean()
        loss_v = self.loss_mse(v.view(-1), v_target)
        loss = loss_p + loss_v
        self.log_dict(
            {
                "loss": loss,
                "policy_loss": loss_p,
                "value_loss": loss_v,
            }
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def main():
    dataset = load_games("games")
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    trainer = L.Trainer(
        max_epochs=5,
        devices=(1 if not torch.cuda.is_available() else torch.cuda.device_count()),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    trainer.fit(Module(), loader)
    os.makedirs("../nets", exist_ok=True)
    torch.save(trainer.model.net.state_dict(), "../nets/gpu_policy.onnx")


if __name__ == "__main__":
    main()
