"""Runtime configuration for the frede web app."""
from __future__ import annotations

from pathlib import Path

# Paths (project root is one level above this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

MODEL_PATH = MODEL_DIR / "fake_review_distilbert"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
TRAINING_META_PATH = MODEL_DIR / "training_meta.json"
HITL_POOL_PATH = DATA_DIR / "hitl_pool.json"
EXAMPLES_PATH = DATA_DIR / "examples.json"
HITL_RESULTS_PATH = DATA_DIR / "hitl_results.csv"

# Model / inference
MODEL_NAME = "distilbert-base-uncased"  # base the fine-tuned checkpoint was made from
MAX_LENGTH = 256
BATCH_SIZE = 16
RANDOM_SEED = 42

# Live XAI budget (tuned for ~3–8 s per request on CPU)
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 150
IG_N_STEPS = 30

# Reason-code thresholds (mirror the notebook)
RC_CONFIDENCE_THRESHOLD = 0.85
