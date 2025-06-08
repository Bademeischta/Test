import argparse
import numpy as np
import torch


def quantize_state_dict(state_dict: dict, scale: int = 1000) -> bytes:
    w1 = state_dict["l1.weight"].t().numpy()
    w2 = state_dict["l2.weight"].t().numpy()
    b2 = state_dict["l2.bias"].numpy()
    w3 = state_dict["out.weight"].squeeze(0).numpy()

    def to_int16(arr):
        arr = np.round(arr * scale)
        return np.clip(arr, -32768, 32767).astype(np.int16)

    w1_q = to_int16(w1)
    w2_q = to_int16(w2)
    b2_q = to_int16(b2)
    w3_q = to_int16(w3)

    return b"".join([
        w1_q.tobytes(),
        w2_q.tobytes(),
        b2_q.tobytes(),
        w3_q.tobytes(),
    ])


def main():
    parser = argparse.ArgumentParser(description="Quantize NNUE weights")
    parser.add_argument("model", help="Path to PyTorch state_dict file")
    parser.add_argument("output", help="Output .nnue file")
    parser.add_argument("--scale", type=int, default=1000, help="Quantization scale factor")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    data = quantize_state_dict(state, args.scale)
    with open(args.output, "wb") as fh:
        fh.write(data)


if __name__ == "__main__":
    main()
