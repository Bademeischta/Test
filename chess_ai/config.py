class Config:
    """Central configuration for the chess AI."""

    # Paths
    CHECKPOINT_DIR = "checkpoints"
    REPLAY_BUFFER_SIZE = 100_000
    GAMES_PER_ITER = 5000
    LOG_DIR = "runs"
    WANDB_PROJECT = "chess-ai"

    # Hardware
    # Always prefer CUDA when available so the network trains on GPUs like the
    # RTX 5070.  A ``torch.device`` object is used to avoid string/device
    # mismatches across the codebase.
    DEVICE = __import__("torch").device(
        "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    )
    SEED = 42

    # MCTS parameters
    # For a single RTX 5070 we use a lighter search to speed up self-play.
    NUM_SIMULATIONS = 400
    C_PUCT = 1.5
    DIRICHLET_EPSILON = 0.25
    DIRICHLET_ALPHA = 0.03

    # Network parameters
    NUM_RES_BLOCKS = 19
    NUM_FILTERS = 256
    # Training hyper parameters tuned for ~8--10 GB VRAM
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 256
    NUM_EPOCHS = 3
    FILTER_QUIET_POSITIONS = True
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
