# SuperEngine Design

## Training Data Format

`selfplay_ray.py` stores each finished game as a PGN file in the `games/` directory.  `train_policy.py` expects these PGN files as input.  For every move in the main line of each game the following tuple is generated:

1. **State** – an `8x8x18` tensor created with `GameEnvironment.encode_board` representing the board before the move.
2. **Policy** – a one‑hot vector of length `ACTION_SIZE` where the played move index (via `move_to_index`) is `1`.
3. **Value** – the final game outcome from the perspective of the player to move at that state (`1` for win, `0` for draw, `-1` for loss).

The resulting tensors are fed to the PyTorch‑Lightning training loop in `train_policy.py`.
