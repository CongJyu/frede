"""Training-time configuration (independent of app/config.py)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
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
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 300  # offline — can afford the notebook default
