#!/usr/bin/env python
"""Minimal training harness for the chess AI."""
import argparse

import torch

from chess_ai.action_index import ACTION_SIZE
from chess_ai.config import Config
from chess_ai.game_environment import GameEnvironment
from chess_ai.network_manager import NetworkManager
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.replay_buffer import ReplayBuffer
from chess_ai.self_play import run_self_play
from chess_ai.trainer import Trainer


def load_or_initialize_network(manager: NetworkManager):
    net = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,
        num_blocks=Config.NUM_RES_BLOCKS,
        filters=Config.NUM_FILTERS,
    ).to(Config.DEVICE)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )
    ckpt = manager.latest_checkpoint()
    if ckpt:
        data = torch.load(ckpt, map_location=Config.DEVICE)
        net.load_state_dict(data["model_state"])
        optimizer.load_state_dict(data["optim_state"])
    return net, optimizer


def main(args):
    manager = NetworkManager()
    net, optimizer = load_or_initialize_network(manager)
    buffer = ReplayBuffer()

    for g in range(args.games):
codex/erweiterungen-für-python--und-c++-komponenten-planen
        for state, policy, value in run_self_play(
            net, num_simulations=args.simulations
        ):

        print(f"Generating game {g + 1}/{args.games}...")
        for state, policy, value in run_self_play(net, num_simulations=args.simulations):
main
            buffer.add(state, policy, value)
    print(f"Collected {len(buffer)} training positions.")

    print("Starting training...")
    trainer = Trainer(net, buffer, optimizer, epochs=args.epochs)
    trainer.train()
    print("Saving checkpoint...")
    manager.save(net, optimizer, "latest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal training loop")
    parser.add_argument(
        "--games", type=int, default=1, help="Number of self-play games"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument(
        "--simulations",
        type=int,
        default=Config.NUM_SIMULATIONS,
        help="MCTS simulations",
    )
    main(parser.parse_args())
