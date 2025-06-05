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

## Building the C++ Engine

The repository also includes a lightweight C++ engine in `superengine/`. Create
a build directory and compile it with CMake:

```bash
mkdir build && cd build
cmake .. && make
```

This produces the `superengine` executable and its libraries.

## Running Unit Tests

After building the project you can run the C++ unit tests from the build
directory using:

```bash
ctest
```

All tests located under `superengine/tests` will be executed.

