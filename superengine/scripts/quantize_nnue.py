import argparse
import sys
import time

import numpy as np
import torch


def _get_param(state_dict: dict, *names: str):
    """Return the first matching parameter tensor from ``state_dict``."""
    for name in names:
        if name in state_dict:
            return state_dict[name]
    raise KeyError(f"None of the parameter names {names} found in state_dict")


def quantize_state_dict(state_dict: dict, scale: int = 1000) -> bytes:
    """Convert model weights to a simple NNUE format.

    This helper is tolerant to different parameter names used by various
    training scripts. It looks for both the current ``fc_*`` names and older
    ``l*``/``out`` style names.
    """

    t0 = time.time()

    keys = list(state_dict.keys())
    print(f"{len(keys)} Keys insgesamt, Beispiel: {keys[:10]}", flush=True)

    print("Exportiere fc_value1.weight...", flush=True)
    w1 = _get_param(state_dict, "fc_value1.weight", "l1.weight").t().numpy()
    print("Exportiere fc_value1.bias...", flush=True)
    b1 = _get_param(state_dict, "fc_value1.bias", "l1.bias").numpy()
    print("Exportiere fc_value2.weight...", flush=True)
    w2 = _get_param(state_dict, "fc_value2.weight", "l2.weight").t().numpy()
    print("Exportiere fc_value2.bias...", flush=True)
    b2 = _get_param(state_dict, "fc_value2.bias", "l2.bias").numpy()
    print("Exportiere fc_policy.weight...", flush=True)
    w3_raw = _get_param(state_dict, "fc_policy.weight", "out.weight")
    w3 = w3_raw.squeeze(0).numpy() if len(w3_raw.shape) == 3 else w3_raw.numpy()
    print("Exportiere fc_policy.bias...", flush=True)
    b3 = _get_param(state_dict, "fc_policy.bias", "out.bias").numpy()

    def to_int16(arr):
        arr = np.round(arr * scale)
        return np.clip(arr, -32768, 32767).astype(np.int16)

    print("Quantisiere...", flush=True)
    w1_q = to_int16(w1)
    b1_q = to_int16(b1)
    w2_q = to_int16(w2)
    b2_q = to_int16(b2)
    w3_q = to_int16(w3)
    b3_q = to_int16(b3)

    parts = [
        w1_q.tobytes(),
        b1_q.tobytes(),
        w2_q.tobytes(),
        b2_q.tobytes(),
        w3_q.tobytes(),
        b3_q.tobytes(),
    ]
    packed = bytearray()
    for idx, part in enumerate(parts, 1):
        print(f"Packe Daten Chunk {idx}/{len(parts)}", flush=True)
        packed.extend(part)
        sys.stdout.flush()

    print("\u2705 Quantisierung und Packen komplett abgeschlossen", flush=True)
    print(f"Quantisierung hat {time.time() - t0:.1f}s gedauert", flush=True)
    return bytes(packed)


def main():
    parser = argparse.ArgumentParser(description="Quantize NNUE weights")
    parser.add_argument("model", help="Path to PyTorch state_dict file")
    parser.add_argument("output", help="Output .nnue file")
    parser.add_argument("--scale", type=int, default=1000, help="Quantization scale factor")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state" in state:
        state = state["model_state"]
    data = quantize_state_dict(state, args.scale)
    with open(args.output, "wb") as fh:
        fh.write(data)


if __name__ == "__main__":
    main()
