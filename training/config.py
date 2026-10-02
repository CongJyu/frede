"""Training-time configuration.

Holds the fine-tuning settings for the archived sentiment-proxy pipeline. The
data paths are imported from `app/config.py` rather than repeated here: they are
read by both the training builders and the serving routes, and two hand-kept
copies of the same path drift apart silently.
"""
from __future__ import annotations

from app.config import (  # noqa: F401  (re-exported for training-side callers)
    DATA_DIR,
    EXAMPLES_PATH,
    HITL_POOL_PATH,
    HITL_RESULTS_PATH,
    MODEL_DIR,
    PROJECT_ROOT,
)

HF_CACHE_DIR = DATA_DIR / "hf_cache"

BASE_MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = MODEL_DIR / "fake_review_distilbert"

# Data
SAMPLE_N = 20_000
VAL_FRAC = 0.30
TEST_FRAC_OF_TEMP = 0.50
RANDOM_SEED = 42

# Training
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
MAX_LENGTH = 256
NUM_THREADS = 10
DEVICE = "mps"  # "cpu" (default) or "mps"

# HITL pool
HITL_N_SAMPLES = 20
HITL_SEED = 123
