import math
from collections import defaultdict

import numpy as np
import torch

from .config import Config


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.P = {}
        self.N = defaultdict(int)
        self.W = defaultdict(float)
        self.Q = defaultdict(float)
        self.is_expanded = False

    def expand(self, policy, legal_moves):
        for move in legal_moves:
            idx = move
            self.P[move] = policy[idx]
            self.N[move] = 0
            self.W[move] = 0.0
            self.Q[move] = 0.0
        self.is_expanded = True

    def select(self, c_puct):
        total_visits = sum(self.N[m] for m in self.P)
        best_move, best_score = None, -float('inf')
        for move in self.P:
            u = self.Q[move] + c_puct * self.P[move] * math.sqrt(total_visits) / (1 + self.N[move])
            if u > best_score:
                best_score = u
                best_move = move
        return best_move


class MCTS:
    def __init__(self, network, c_puct: float = Config.C_PUCT, num_simulations: int = Config.NUM_SIMULATIONS):
        self.network = network.to(Config.DEVICE)
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run(self, root_state):
        root = MCTSNode(root_state)
        state_tensor = torch.tensor(root_state, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            log_p, v = self.network(state_tensor)
        policy = torch.exp(log_p[0]).cpu().numpy()
        root.expand(policy, range(len(policy)))
        moves = list(root.P.keys())
        if moves:
            noise = np.random.dirichlet([Config.DIRICHLET_ALPHA] * len(moves))
            for idx, m in enumerate(moves):
                root.P[m] = (1 - Config.DIRICHLET_EPSILON) * root.P[m] + Config.DIRICHLET_EPSILON * noise[idx]

        for _ in range(self.num_simulations):
            node = root
            search_path = []
            while node.is_expanded:
                move = node.select(self.c_puct)
                search_path.append((node, move))
                if move not in node.children:
                    break
                node = node.children[move]
            # evaluation
            if not node.is_expanded:
                state_tensor = torch.tensor(node.state, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
                with torch.no_grad():
                    log_p, v = self.network(state_tensor)
                policy = torch.exp(log_p[0]).cpu().numpy()
                node.expand(policy, range(len(policy)))
            value = v.item()
            for parent, move in reversed(search_path):
                parent.N[move] += 1
                parent.W[move] += value
                parent.Q[move] = parent.W[move] / parent.N[move]
                value = -value
        return {m: root.N[m] for m in root.P}
