import math
from collections import defaultdict

import numpy as np
import torch
import chess

from .config import Config
from .game_environment import GameEnvironment
from .action_index import move_to_index


class MCTSNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.P = {}
        self.N = defaultdict(int)
        self.W = defaultdict(float)
        self.Q = defaultdict(float)
        self.is_expanded = False

    def expand(self, policy, legal_moves):
        for move in legal_moves:
            idx = move_to_index(move)
            self.P[move] = policy[idx]
            self.N[move] = 0
            self.W[move] = 0.0
            self.Q[move] = 0.0
        self.is_expanded = True

    def select(self, c_puct):
        total_visits = sum(self.N[m] for m in self.P)
        best_move, best_score = None, -float('inf')
        for move in self.P:
            u = (
                self.Q[move]
                + c_puct
                * self.P[move]
                * math.sqrt(total_visits)
                / (1 + self.N[move])
            )
            if u > best_score:
                best_score = u
                best_move = move
        return best_move


class MCTS:
    def __init__(
        self,
        network,
        c_puct: float = Config.C_PUCT,
        num_simulations: int = Config.NUM_SIMULATIONS,
    ):
        self.network = network.to(Config.DEVICE)
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run(self, root_board: chess.Board):
        root = MCTSNode(root_board.copy())
        state_tensor = torch.tensor(
            GameEnvironment.encode_board(root.board),
            dtype=torch.float32,
            device=Config.DEVICE,
        ).unsqueeze(0)
        with torch.no_grad():
            log_p, v = self.network(state_tensor)
        policy = torch.exp(log_p[0]).cpu().numpy()
        root.expand(policy, list(root.board.legal_moves))
        moves = list(root.P.keys())
        if moves:
            noise = np.random.dirichlet([Config.DIRICHLET_ALPHA] * len(moves))
            for idx, m in enumerate(moves):
                root.P[m] = (
                    (1 - Config.DIRICHLET_EPSILON) * root.P[m]
                    + Config.DIRICHLET_EPSILON * noise[idx]
                )

        for _ in range(self.num_simulations):
            node = root
            search_path = []
            while node.is_expanded and node.P:
                move = node.select(self.c_puct)
                search_path.append((node, move))
                if move not in node.children:
                    node.board.push(move)
                    child_board = node.board.copy()
                    node.board.pop()
                    node.children[move] = MCTSNode(child_board, parent=node)
                node = node.children[move]

            state_tensor = torch.tensor(
                GameEnvironment.encode_board(node.board),
                dtype=torch.float32,
                device=Config.DEVICE,
            ).unsqueeze(0)
            with torch.no_grad():
                log_p, v = self.network(state_tensor)
            policy = torch.exp(log_p[0]).cpu().numpy()
            node.expand(policy, list(node.board.legal_moves))
            value = v.item()
            for parent, move in reversed(search_path):
                parent.N[move] += 1
                parent.W[move] += value
                parent.Q[move] = parent.W[move] / parent.N[move]
                value = -value
        return {move_to_index(m): root.N[m] for m in root.P}
