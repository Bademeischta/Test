# SuperEngine Design

## Engine Components

The engine is split into a few small C++ modules under `engine/`:

- **bitboard** – utilities for manipulating 64‑bit board representations.
- **movegen** – generates pseudo legal moves from a given position.
- **search** – implements a very small alpha‑beta search routine.
- **uci** – thin wrapper providing a UCI compliant command loop.

These pieces are compiled into the `engine` library which is linked by the
`superengine` executable.

## NNUE Integration

Evaluation is provided by a minimal NNUE network defined in `nnue_eval.*`. The
engine extracts features from the current position, feeds them through the
network and returns the resulting score during search. Network weights can be
loaded at runtime from the `nets/` directory.

## Training Workflow

Training data is produced through self‑play using the Python tools in the root
`chess_ai` package. Games are stored and converted to feature matrices with
`prepare_data.py`. The neural network weights are optimized using the scripts
`train_policy.py` for the policy network and `train_nnue.py` for the NNUE
weights. Once trained, the NNUE parameters can be loaded by the engine for play.
