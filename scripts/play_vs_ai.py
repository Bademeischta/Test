#!/usr/bin/env python
"""Play a game against the latest trained network."""
import argparse
import chess
import torch

from chess_ai.config import Config
from chess_ai.game_environment import GameEnvironment
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.action_index import ACTION_SIZE, index_to_move
from chess_ai.mcts import MCTS
from chess_ai.network_manager import NetworkManager


def load_network(manager: NetworkManager) -> PolicyValueNet:
    net = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,
        num_blocks=Config.NUM_RES_BLOCKS,
        filters=Config.NUM_FILTERS,
    ).to(Config.DEVICE)
    ckpt = manager.latest_checkpoint()
    if not ckpt:
        raise FileNotFoundError("No trained network found in checkpoints")
    data = torch.load(ckpt, map_location=Config.DEVICE)
    net.load_state_dict(data["model_state"])
    net.eval()
    return net


def main(args):
    manager = NetworkManager()
    net = load_network(manager)
    env = GameEnvironment()
    ai_color = chess.BLACK if args.play_white else chess.WHITE
    while True:
        print(env.board)
        if env.board.turn == ai_color:
            mcts = MCTS(net, num_simulations=args.simulations)
            visits = mcts.run(env.board)
            best_idx = max(visits, key=visits.get)
            move = index_to_move(best_idx)
            print(f"AI plays: {move.uci()}")
        else:
            move_uci = input("Your move: ")
            move = chess.Move.from_uci(move_uci)
            if move not in env.board.legal_moves:
                print("Illegal move, try again")
                continue
        _, _, done = env.step(move)
        if done:
            print(env.board)
            print("Game over:", env.board.result())
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against the trained AI")
    parser.add_argument("--play-white", action="store_true", help="Play as white instead of black")
    parser.add_argument("--simulations", type=int, default=Config.NUM_SIMULATIONS, help="MCTS simulations for AI moves")
    main(parser.parse_args())
