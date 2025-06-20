import numpy as np

from prometheus_client import Counter

from .game_environment import GameEnvironment
from .mcts import MCTS
from .config import Config
from .action_index import ACTION_SIZE, index_to_move

# Metric to count completed self-play games
self_play_games_total = Counter(
    "self_play_games_total", "Number of self-play games generated"
)


def run_self_play(network, num_simulations: int = Config.NUM_SIMULATIONS):
    """Generate self-play data from games played by the network."""
    env = GameEnvironment()
    network = network.to(Config.DEVICE)
    mcts = MCTS(network, num_simulations=num_simulations)
    state = env.reset()
    trajectory = []
    current_player = 1
    move_number = 0
    while True:
        visit_counts = mcts.run(env.board)
        counts = np.array([visit_counts[m] for m in visit_counts], dtype=np.float32)
        move_indices = list(visit_counts.keys())
        temperature = 1.0 if move_number < 30 else 0.1
        probs = counts ** (1.0 / temperature)
        probs /= probs.sum()
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        for idx, p in zip(move_indices, probs):
            pi[idx] = p
        chosen = np.random.choice(len(move_indices), p=probs)
        best_move_idx = move_indices[chosen]
        move = index_to_move(best_move_idx)
        is_quiet = env.is_quiet_move(move)
        if not Config.FILTER_QUIET_POSITIONS or is_quiet:
            trajectory.append((state, pi, current_player))
        state, reward, done = env.step(move)
        if done:
            for s, p, player in trajectory:
                z = reward if player == current_player else -reward
                yield s, p, z
            self_play_games_total.inc()
            break
        current_player *= -1
        move_number += 1


if __name__ == "__main__":
    """Run a short self-play demo when executed as a script."""
    from .policy_value_net import PolicyValueNet

    net = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,
        num_blocks=1,
        filters=32,
    )

    for i, (_, _, value) in enumerate(run_self_play(net, num_simulations=1)):
        print(f"Step {i}: value={value:.2f}")
        if i >= 2:
            break

    print("Self-play finished successfully.")
