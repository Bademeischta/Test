import chess

from .game_environment import GameEnvironment
from .mcts import MCTS
from .config import Config
from .action_index import index_to_move


def evaluate(
    net_a,
    net_b,
    num_games: int = 10,
    num_simulations: int = Config.NUM_SIMULATIONS,
    max_moves: int | None = None,
):
    """Play ``num_games`` games between ``net_a`` and ``net_b``.

    Parameters
    ----------
    net_a, net_b : torch.nn.Module
        Networks to pit against each other.
    num_games : int, optional
        Number of games to play.
    num_simulations : int, optional
        MCTS simulations per move.
    max_moves : int or None, optional
        If given, terminate games after this many half-moves (plies) and
        count them as draws.
    """

    stats = {"wins": 0, "losses": 0, "draws": 0}
    for g in range(num_games):
        env = GameEnvironment()
        current_player = 1
        nets = [net_a.to(Config.DEVICE), net_b.to(Config.DEVICE)]
        move_counter = 0
        while True:
            net = nets[0] if env.board.turn == chess.WHITE else nets[1]
            mcts = MCTS(net, num_simulations=num_simulations)
            visit_counts = mcts.run(env.board)
            best_move_idx = max(visit_counts, key=visit_counts.get)
            move = index_to_move(best_move_idx)
            _, reward, done = env.step(move)
            move_counter += 1
            if not done and (max_moves is not None and move_counter >= max_moves):
                reward = 0  # treat as draw when exceeding move limit
                done = True
            if done:
                if reward == 1:
                    stats["wins"] += 1 if env.board.turn == chess.BLACK else 0
                    stats["losses"] += 1 if env.board.turn == chess.WHITE else 0
                elif reward == -1:
                    stats["losses"] += 1 if env.board.turn == chess.BLACK else 0
                    stats["wins"] += 1 if env.board.turn == chess.WHITE else 0
                else:
                    stats["draws"] += 1
                break
            current_player *= -1
    return stats
