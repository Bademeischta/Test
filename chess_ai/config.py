class Config:
    """Central configuration for the chess AI."""

    # Paths
    CHECKPOINT_DIR = "checkpoints"
    REPLAY_BUFFER_SIZE = 100_000

    # Hardware
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    SEED = 42

    # MCTS parameters
    NUM_SIMULATIONS = 800
    C_PUCT = 1.5
    DIRICHLET_EPSILON = 0.25
    DIRICHLET_ALPHA = 0.03

    # Network parameters
    NUM_RES_BLOCKS = 19
    NUM_FILTERS = 256
    LEARNING_RATE = 0.01
    BATCH_SIZE = 32
    FILTER_QUIET_POSITIONS = True
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
