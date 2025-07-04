#!/usr/bin/env python
"""Minimal training harness for the chess AI."""


import os
import sys
import subprocess



import argparse
import torch

from chess_ai.config import Config
from chess_ai.game_environment import GameEnvironment
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.replay_buffer import ReplayBuffer
from chess_ai.self_play import run_self_play
from chess_ai.trainer import Trainer
from chess_ai.action_index import ACTION_SIZE

from chess_ai.network_manager import NetworkManager, _unwrap
from chess_ai.evaluation import evaluate

from chess_ai.network_manager import NetworkManager



def load_or_initialize_network(manager: NetworkManager):
    net = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,
        num_blocks=Config.NUM_RES_BLOCKS,
        filters=Config.NUM_FILTERS,
    ).to(Config.DEVICE)
    lr = float(getattr(Config, "LEARNING_RATE", 1e-3))
    weight_decay = float(getattr(Config, "WEIGHT_DECAY", 0.0))
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=Config.MOMENTUM,
        weight_decay=weight_decay,
    )
    ckpt = manager.latest_checkpoint()
    if ckpt:

        manager.load(ckpt, net, optimizer)
    return net, optimizer


def sprt(w, l, d, elo0=-5, elo1=5, alpha=0.05, beta=0.05):
    """Simple sequential probability ratio test for Elo."""
    import math

    p0 = 1 / (1 + 10 ** (elo0 / 400))
    p1 = 1 / (1 + 10 ** (elo1 / 400))
    llr = (
        w * math.log(p1 / p0)
        + l * math.log((1 - p1) / (1 - p0))
        + d * math.log(0.5 / 0.5)
    )
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    if llr >= A:
        return True
    if llr <= B:
        return False
    return None


def main(args):
    manager = NetworkManager()
    old_ckpt = manager.latest_checkpoint()

        data = torch.load(ckpt, map_location=Config.DEVICE)
        net.load_state_dict(data["model_state"])
        optimizer.load_state_dict(data["optim_state"])
    return net, optimizer


def main(args):
    manager = NetworkManager()

    net, optimizer = load_or_initialize_network(manager)
    buffer = ReplayBuffer()

    for g in range(args.games):

        print(f"Generating game {g + 1}/{args.games}...")
        for state, policy, value in run_self_play(
            net, num_simulations=args.simulations
        ):
            buffer.add(state, policy, value)
    print(f"Collected {len(buffer)} training positions.")

    print("Starting training...")
    trainer = Trainer(net, buffer, optimizer, epochs=args.epochs)
    trainer.train()

    new_ckpt = manager.save(net, optimizer, "latest")
    try:
        subprocess.run(
            [
                "python",
                "superengine/scripts/quantize_nnue.py",
                new_ckpt,
                "superengine/nets/iter_latest.nnue",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Quantization failed: {exc}")

    if old_ckpt:
        old_net = PolicyValueNet(
            GameEnvironment.NUM_CHANNELS,
            ACTION_SIZE,
            num_blocks=Config.NUM_RES_BLOCKS,
            filters=Config.NUM_FILTERS,
        ).to(Config.DEVICE)
        data = torch.load(old_ckpt, map_location=Config.DEVICE)
        old_net.load_state_dict(data["model_state"])
        stats = evaluate(
            net,
            old_net,
            num_games=1000,
            num_simulations=Config.NUM_SIMULATIONS,
            max_moves=60,
        )
        print("Eval stats", stats)
        if sprt(stats["wins"], stats["losses"], stats["draws"]):
            print("New network accepted")
        else:
            print("New network rejected, reverting")
            manager.load(old_ckpt, net, optimizer)

    os.makedirs("nets", exist_ok=True)

    # Build a fresh PolicyValueNet without checkpointing/ORTModule for export
    export_model = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,
        num_blocks=Config.NUM_RES_BLOCKS,
        filters=Config.NUM_FILTERS,
    )

    # Load the trained weights from the wrapped model
    base = _unwrap(net)
    export_model.load_state_dict(base.state_dict())

    export_model.eval()
    export_model.to("cpu")
    dummy_input = torch.randn(1, 18, 8, 8, device="cpu")

    torch.onnx.export(
        export_model,
        dummy_input,
        "nets/final_model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
    )
    print("✅ ONNX-Export abgeschlossen: nets/final_model.onnx")

        for state, policy, value in run_self_play(net, num_simulations=args.simulations):
            buffer.add(state, policy, value)
    trainer = Trainer(net, buffer, optimizer, epochs=args.epochs)
    trainer.train()
    manager.save(net, optimizer, "latest")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal training loop")

    parser.add_argument(
        "--games",
        type=int,
        default=Config.GAMES_PER_ITER,
        help="Number of self-play games",
    )
    parser.add_argument(
        "--epochs", type=int, default=Config.NUM_EPOCHS, help="Training epochs"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=Config.NUM_SIMULATIONS,
        help="MCTS simulations",
    )
    args = parser.parse_args()

    main(args)

    parser.add_argument("--games", type=int, default=1, help="Number of self-play games")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--simulations", type=int, default=Config.NUM_SIMULATIONS, help="MCTS simulations")
    main(parser.parse_args())

