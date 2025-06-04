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
pip install torch python-chess
python -m chess_ai.self_play
python -m chess_ai.evaluation
```

Adjust hyperparameters in `chess_ai/config.py` to tune the behaviour.

