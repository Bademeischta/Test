#!/usr/bin/env python3
"""
Quantize PyTorch state_dict into a simple NNUE .nnue binary format.

Usage:
    python quantize_nnue.py path/to/model.pt path/to/output.nnue [--scale SCALE]

This script reads the .pt checkpoint, extracts the required FC and policy parameters,
quantizes them to int16 with a scale factor, and writes the concatenated binary to
an output .nnue file. The input .pt file is never modified.
"""
import argparse
import sys
import time
import os

import numpy as np
import torch


def _get_param(state_dict: dict, *names: str):
    """Return the first matching tensor from state_dict under any of the provided names."""
    for name in names:
        if name in state_dict:
            return state_dict[name]
    raise KeyError(f"None of the parameter names {names} found in state_dict")


def quantize_state_dict(state_dict: dict, scale: int = 1000) -> bytes:
    """Convert selected parameters to int16 and pack into a single bytes object."""
    t0 = time.time()

    keys = list(state_dict.keys())
    print(f"{len(keys)} Keys insgesamt, Beispiel: {keys[:10]}", flush=True)

    # Extract and transpose FC1
    print("Exportiere fc_value1.weight...", flush=True)
    w1 = _get_param(state_dict, "fc_value1.weight", "l1.weight").t().cpu().numpy()
    print("Exportiere fc_value1.bias...", flush=True)
    b1 = _get_param(state_dict, "fc_value1.bias", "l1.bias").cpu().numpy()

    # Extract and transpose FC2
    print("Exportiere fc_value2.weight...", flush=True)
    w2 = _get_param(state_dict, "fc_value2.weight", "l2.weight").t().cpu().numpy()
    print("Exportiere fc_value2.bias...", flush=True)
    b2 = _get_param(state_dict, "fc_value2.bias", "l2.bias").cpu().numpy()

    # Extract policy layer weights
    print("Exportiere fc_policy.weight...", flush=True)
    w3_raw = _get_param(state_dict, "fc_policy.weight", "out.weight")
    w3 = w3_raw.squeeze(0).cpu().numpy() if w3_raw.ndim == 3 else w3_raw.cpu().numpy()
    print("Exportiere fc_policy.bias...", flush=True)
    b3 = _get_param(state_dict, "fc_policy.bias", "out.bias").cpu().numpy()

    # Quantization helper
    def to_int16(arr: np.ndarray) -> np.ndarray:
        arr = np.round(arr * scale)
        return np.clip(arr, -32768, 32767).astype(np.int16)

    print("Quantisiere...", flush=True)
    w1_q = to_int16(w1); b1_q = to_int16(b1)
    w2_q = to_int16(w2); b2_q = to_int16(b2)
    w3_q = to_int16(w3); b3_q = to_int16(b3)

    parts = [w1_q.tobytes(), b1_q.tobytes(), w2_q.tobytes(), b2_q.tobytes(), w3_q.tobytes(), b3_q.tobytes()]
    packed = bytearray()
    total = len(parts)
    for idx, part in enumerate(parts, 1):
        print(f"Packe Daten Chunk {idx}/{total}", flush=True)
        packed.extend(part)

    print("✅ Quantisierung und Packen komplett abgeschlossen", flush=True)
    print(f"Quantisierung hat {time.time() - t0:.1f}s gedauert", flush=True)
    return bytes(packed)


def main():
    parser = argparse.ArgumentParser(description="Quantize NNUE weights from PyTorch .pt checkpoint.")
    parser.add_argument("model", help="Path to PyTorch checkpoint (.pt) file")
    parser.add_argument("output", help="Destination .nnue output file")
    parser.add_argument("--scale", type=int, default=1000, help="Quantization scale factor")
    args = parser.parse_args()

    # Validate paths
    if not os.path.isfile(args.model):
        print(f"Fehler: Modell-Datei nicht gefunden: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not args.model.endswith('.pt'):
        print("Warnung: Eingabedatei ist keine .pt", file=sys.stderr)

    # Load checkpoint state dict
    try:
        state = torch.load(args.model, map_location='cpu')
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints: {e}", file=sys.stderr)
        sys.exit(1)

    if 'state_dict' in state:
        state = state['state_dict']
    elif 'model_state' in state:
        state = state['model_state']

    # Perform quantization
    data = quantize_state_dict(state, args.scale)

    # Write output file
    try:
        with open(args.output, 'wb') as fh:
            fh.write(data)
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Output geschrieben: {args.output}")


if __name__ == "__main__":
    main()
