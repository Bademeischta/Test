import numpy as np
import torch

from .game_environment import GameEnvironment
from .mcts import MCTS
from .config import Config
from .action_index import ACTION_SIZE, index_to_move


def run_self_play(network, num_simulations: int = Config.NUM_SIMULATIONS):
    """Generate self-play data from games played by the network."""
    env = GameEnvironment()
    network = network.to(Config.DEVICE)
    mcts = MCTS(network, num_simulations=num_simulations)
    state = env.reset()
    trajectory = []
    current_player = 1
    while True:
        visit_counts = mcts.run(env.board)
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        for idx, c in visit_counts.items():
            pi[idx] = c
        best_move_idx = max(visit_counts, key=visit_counts.get)
        move = index_to_move(best_move_idx)
        trajectory.append((state, pi, current_player))
        state, reward, done = env.step(move)
        if done:
            for s, p, player in trajectory:
                z = reward if player == current_player else -reward
                yield s, p, z
            break
        current_player *= -1
