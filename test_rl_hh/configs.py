"""Hyperparameters for RL-HH Dueling DDQN training."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
INPUT_DIM = 33
N_TOOLS = 5

# Grouped feature slices (indices into the 33-dim observation)
ROASTER_FEATURES = slice(1, 16)       # 15: 5 status + 5 timer + 5 last_sku
INVENTORY_FEATURES = slice(16, 27)    # 11: RC, mto, pipeline, GC, restock
CONTEXT_FEATURES_IDX = [0] + list(range(27, 33))  # time + 6 one-hot = 7

ROASTER_DIM = 15
INVENTORY_DIM = 11
CONTEXT_DIM = 7

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LR = 5e-4
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 50_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_FRAC = 0.30
TAU = 0.005            # soft target update rate
NUM_EPISODES = 300_000
TRAIN_EVERY = 4        # train once every N decisions (inside fast_loop)
TRAINS_PER_EP = 4      # gradient steps per episode (keep low for speed)
GRAD_CLIP = 10.0       # max gradient norm

# ---------------------------------------------------------------------------
# UPS — always read from Input_data/shift_parameters.csv at runtime.
# These are ONLY used as fallbacks if data loading somehow fails.
# ---------------------------------------------------------------------------
UPS_LAMBDA_FALLBACK = 3
UPS_MU_FALLBACK = 20

# ---------------------------------------------------------------------------
# Logging / checkpoints
# ---------------------------------------------------------------------------
LOG_INTERVAL = 1_000
CHECKPOINT_INTERVAL = 50_000

# ---------------------------------------------------------------------------
# Tool names (for logging / display)
# ---------------------------------------------------------------------------
TOOL_NAMES = ["PSC_THROUGHPUT", "GC_RESTOCK", "MTO_DEADLINE", "SETUP_AVOID", "WAIT"]
