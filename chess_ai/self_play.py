import numpy as np
import torch

from .game_environment import GameEnvironment
from .mcts import MCTS


def run_self_play(network, num_simulations=50):
    env = GameEnvironment()
    mcts = MCTS(network, num_simulations=num_simulations)
    state = env.reset()
    trajectory = []
    current_player = 1
    while True:
        visit_counts = mcts.run(state)
        pi = np.array(list(visit_counts.values()), dtype=np.float32)
        best_move_idx = max(visit_counts, key=visit_counts.get)
        move = list(env.legal_moves())[best_move_idx]
        trajectory.append((state, pi, current_player))
        state, reward, done = env.step(move)
        if done:
            for s, p, player in trajectory:
                z = reward if player == current_player else -reward
                yield s, p, z
            break
        current_player *= -1
