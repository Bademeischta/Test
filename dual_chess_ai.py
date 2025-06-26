import argparse
import random
from typing import List, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Required versions:
# python-chess==1.999
# torch==2.1.0
# numpy==1.24.0
# h5py==3.8.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SIZE = 4672  # AlphaZero style move encoding


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode board into an 8x8x20 tensor."""
    planes = np.zeros((20, 8, 8), dtype=np.float32)
    pieces = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for sq, pc in board.piece_map().items():
        r, c = divmod(sq, 8)
        offset = 0 if pc.color == chess.WHITE else 6
        planes[offset + pieces[pc.piece_type]][r][c] = 1
    planes[12][:] = int(board.has_kingside_castling_rights(chess.WHITE))
    planes[13][:] = int(board.has_queenside_castling_rights(chess.WHITE))
    planes[14][:] = int(board.has_kingside_castling_rights(chess.BLACK))
    planes[15][:] = int(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        planes[16][r][c] = 1
    planes[17][:] = int(board.turn)
    planes[18][:] = board.halfmove_clock / 50.0
    planes[19][:] = board.fullmove_number / 100.0
    return planes


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """Policy/Value network for self-play agent."""

    def __init__(self, blocks: int = 10, filters: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(20, filters, 3, padding=1)
        self.bn = nn.BatchNorm2d(filters)
        self.res = nn.ModuleList([ResidualBlock(filters) for _ in range(blocks)])
        self.p_conv = nn.Conv2d(filters, 32, 3, padding=1)
        self.p_fc = nn.Linear(32 * 8 * 8, ACTION_SIZE)
        self.v_conv = nn.Conv2d(filters, 3, 3, padding=1)
        self.v_fc1 = nn.Linear(3 * 8 * 8, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res:
            x = block(x)
        p = F.relu(self.p_conv(x))
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.p_fc(p), dim=1)
        v = F.relu(self.v_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v.squeeze(1)


class DQNNet(nn.Module):
    """Dueling DQN network."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(20, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.res = nn.ModuleList([ResidualBlock(channels) for _ in range(5)])
        self.adv_conv = nn.Conv2d(channels, 32, 3, padding=1)
        self.val_conv = nn.Conv2d(channels, 32, 3, padding=1)
        self.adv_fc = nn.Linear(32 * 8 * 8, ACTION_SIZE)
        self.val_fc = nn.Linear(32 * 8 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res:
            x = block(x)
        adv = F.relu(self.adv_conv(x)).view(x.size(0), -1)
        val = F.relu(self.val_conv(x)).view(x.size(0), -1)
        adv = self.adv_fc(adv)
        val = self.val_fc(val)
        return val + adv - adv.mean(1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List = []
        self.pos = 0

    def add(self, *transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class SelfPlayAgent:
    def __init__(self, net: AlphaZeroNet, simulations: int = 800):
        from chess_ai.mcts import MCTS

        self.net = net
        self.simulations = simulations
        self.mcts_class = MCTS

    def select_move(self, board: chess.Board) -> chess.Move:
        mcts = self.mcts_class(self.net, c_puct=4.0, num_simulations=self.simulations)
        visits = mcts.run(board)
        best_idx = max(visits, key=visits.get)
        from chess_ai.action_index import index_to_move

        return index_to_move(best_idx)


class QLearningAgent:
    def __init__(self, net: DQNNet, target: DQNNet, buffer: ReplayBuffer):
        self.net = net
        self.target = target
        self.buffer = buffer
        self.gamma = 0.99
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.steps = 0

    def select_move(self, board: chess.Board) -> chess.Move:
        s = torch.tensor(encode_board(board), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q = self.net(s)
        idx = int(q.argmax())
        from chess_ai.action_index import index_to_move

        move = index_to_move(idx)
        if move not in board.legal_moves:
            move = random.choice(list(board.legal_moves))
        return move

    def update(self, batch_size: int = 64):
        if len(self.buffer) < batch_size:
            return
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        s = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        r = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        s_next = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
        d = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        q = self.net(s)
        next_q = self.target(s_next).max(1)[0]
        target = r + self.gamma * next_q * (1 - d)
        current = q.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(current, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        if self.steps % 1000 == 0:
            self.target.load_state_dict(self.net.state_dict())
        self.steps += 1


class DualChessAI:
    def __init__(self, mix_ratio: float = 0.5):
        self.sp_net = AlphaZeroNet().to(DEVICE)
        self.dqn_net = DQNNet().to(DEVICE)
        self.dqn_target = DQNNet().to(DEVICE)
        self.dqn_target.load_state_dict(self.dqn_net.state_dict())
        self.buffer = ReplayBuffer()
        self.sp_agent = SelfPlayAgent(self.sp_net)
        self.dqn_agent = QLearningAgent(self.dqn_net, self.dqn_target, self.buffer)
        self.mix_ratio = mix_ratio

    def play_game(self) -> None:
        board = chess.Board()
        states = []
        actions = []
        while not board.is_game_over(claim_draw=True) and len(states) < 200:
            if random.random() < self.mix_ratio:
                move = self.sp_agent.select_move(board)
            else:
                move = self.dqn_agent.select_move(board)
            state = encode_board(board)
            board.push(move)
            states.append(state)
            actions.append(move)
        result = board.result()
        reward = 0.0
        if result == "1-0":
            reward = 1.0
        elif result == "0-1":
            reward = -1.0
        for s in states:
            next_state = encode_board(board)
            done = board.is_game_over()
            self.buffer.add(s, 0, reward, next_state, float(done))

    def train(self, episodes: int = 10):
        for ep in range(episodes):
            self.play_game()
            self.dqn_agent.update()
            print(f"Episode {ep+1}/{episodes} - buffer size {len(self.buffer)}")

    def uci_loop(self):
        board = chess.Board()
        print("id name DualChessAI")
        print("uciok")
        while True:
            cmd = input().strip()
            if cmd == "isready":
                print("readyok")
            elif cmd.startswith("position"):
                if "startpos" in cmd:
                    board = chess.Board()
                elif "fen" in cmd:
                    fen = cmd.split("fen")[-1].strip()
                    board = chess.Board(fen)
            elif cmd.startswith("go"):
                move = self.sp_agent.select_move(board)
                board.push(move)
                print(f"bestmove {move.uci()}")
            elif cmd == "quit":
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    args = parser.parse_args()
    ai = DualChessAI()
    if args.mode == "train":
        ai.train(episodes=1)
    else:
        ai.uci_loop()
