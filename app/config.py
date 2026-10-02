"""Runtime configuration for the frede web app.

Only entries the running app actually reads live here. Settings that used to
support the archived sentiment pipeline (its model paths, LIME and Integrated
Gradients budgets) have been removed — a config full of knobs that do nothing
reads as tunable when it is not.

`training/config.py` re-exports the shared data paths from this module, so the
two cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

# Paths (project root is one level above this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
# Built React + Ant Design SPA (Vite output). Build with `make frontend`.
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

# The detector: a gradient-boosted classifier over GPT-2 surprisal + stylometric
# features, plus the human/machine reference profile the reason codes quote
# percentiles against. Both are produced by `make stage2`; the app cannot start
# without them and says so through /api/health.
DETECTOR_PATH = MODEL_DIR / "stage2_detector.joblib"
REFERENCE_PATH = MODEL_DIR / "stage2_reference.json"

# The reference LM whose surprisal the detector reads. Fixed at training time —
# changing it invalidates the saved model, since every surprisal feature shifts.
SURPRISAL_LM = "gpt2"
SURPRISAL_MAX_LENGTH = 256

# Stage 4 artifacts, written by `make stage4` and read by the HITL and Examples
# routes.
HITL_POOL_PATH = DATA_DIR / "hitl_pool.json"
EXAMPLES_PATH = DATA_DIR / "examples.json"
HITL_RESULTS_PATH = DATA_DIR / "hitl_results.csv"

# Reason-code threshold. A position within the *human* review distribution, not
# an absolute value: a real review is marked unusual only when it falls outside
# the range real reviewers occupy. The cut is deliberately wide (5th/95th
# percentile) so a reason code means a genuine outlier rather than ordinary
# variation between reviewers.
RC_PERCENTILE_CUT = 5.0

# The operating threshold is NOT here — it is calibrated at fit time and shipped
# inside DETECTOR_PATH, because a hard-coded value would carry no false-positive
# guarantee. Read it from the loaded detector (`DetectorService.threshold`).
