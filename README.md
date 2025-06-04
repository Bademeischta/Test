# Chess AI Skeleton

This repository contains a minimal Python skeleton for a self-learning chess engine.
The code implements basic components described in the included specification:

- **GameEnvironment** using `python-chess` for rule handling and board encoding
- **PolicyValueNet** with a residual convolutional architecture
- **MCTS** search using neural network priors and value estimates
- **ReplayBuffer** to store training examples
- **Trainer** performing simple policy/value updates
- **Self-play** routine for data generation

This code is only a starting point and omits many details required for a
production-ready system, but demonstrates the overall structure.

## Usage

Install requirements and run a short self-play session:

```bash
pip install -r requirements.txt
# start self‑play to populate the replay buffer
python -m chess_ai.self_play

# train the network from the gathered data
python -m chess_ai.trainer

# evaluate the latest model against the previously best one
python -m chess_ai.evaluate
```

All hyperparameters reside in `chess_ai/config.py`.  Adjust them to tune the
behaviour of the engine or to change file locations for checkpoints and the
replay buffer.

