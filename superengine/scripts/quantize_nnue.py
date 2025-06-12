import argparse
import numpy as np
import torch
from tqdm.auto import tqdm


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

    print("Ver\xC3\xBCgbare Schl\xC3\xBCssel im state_dict:", list(state_dict.keys()))

    print("Exportiere fc_value1.weight...")
    w1 = _get_param(state_dict, "fc_value1.weight", "l1.weight").t().numpy()
    print("Exportiere fc_value1.bias...")
    b1 = _get_param(state_dict, "fc_value1.bias", "l1.bias").numpy()
    print("Exportiere fc_value2.weight...")
    w2 = _get_param(state_dict, "fc_value2.weight", "l2.weight").t().numpy()
    print("Exportiere fc_value2.bias...")
    b2 = _get_param(state_dict, "fc_value2.bias", "l2.bias").numpy()
    print("Exportiere fc_policy.weight...")
    w3_raw = _get_param(state_dict, "fc_policy.weight", "out.weight")
    w3 = w3_raw.squeeze(0).numpy() if len(w3_raw.shape) == 3 else w3_raw.numpy()
    print("Exportiere fc_policy.bias...")
    b3 = _get_param(state_dict, "fc_policy.bias", "out.bias").numpy()

    def to_int16(arr):
        arr = np.round(arr * scale)
        return np.clip(arr, -32768, 32767).astype(np.int16)

    print("Quantisiere...")
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
    for part in tqdm(parts, desc="Packe Daten", unit="chunk"):
        packed.extend(part)
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
