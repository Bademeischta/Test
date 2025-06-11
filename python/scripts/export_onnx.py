#!/usr/bin/env python3
import torch
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.game_environment import GameEnvironment
from chess_ai.action_index import ACTION_SIZE


def main(ckpt_path: str, onnx_out: str):
    # 1) Model reconstruction
    model = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # 2) Dummy input (Batch=1, Channels=18, 8x8)
    dummy = torch.zeros(1, GameEnvironment.NUM_CHANNELS, 8, 8, dtype=torch.float32)

    # 3) Export
    torch.onnx.export(
        model,
        dummy,
        onnx_out,
        input_names=["input"],
        output_names=["policy", "value"],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"\u2713  ONNX model written: {onnx_out}")


if __name__ == "__main__":
    import argparse
    import pathlib

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="policy.onnx")
    args = p.parse_args()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    main(args.ckpt, args.out)
