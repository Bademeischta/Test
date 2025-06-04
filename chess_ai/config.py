class Config:
    """Central configuration for the chess AI."""

    # Paths
    CHECKPOINT_DIR = "checkpoints"
    REPLAY_BUFFER_PATH = "data/replay_buffer.h5"
    LOG_DIR = "logs"
    REPLAY_BUFFER_SIZE = 100_000
    NUM_CHANNELS = 18

    # Hardware
    DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    SEED = 42

    # MCTS parameters
    NUM_SIMULATIONS = 800
    C_PUCT = 1.5
    DIRICHLET_EPSILON = 0.25
    DIRICHLET_ALPHA = 0.03

    # Self-play
    RELOAD_INTERVAL = 100

    # Network parameters
    NUM_RES_BLOCKS = 19
    NUM_FILTERS = 256
    LEARNING_RATE = 0.01
    BATCH_SIZE = 32
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_WARMUP_STEPS = 1000
    LR_DECAY_TYPE = "cosine"

    # Evaluation
    EVALUATION_GAMES = 50

