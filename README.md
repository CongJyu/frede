# Frede — Fake Restaurant Review Detector (web app)

A Python web app port of `ref/guided_study.ipynb` ("Fake Restaurant Review
Detection and XAI Analysis"). It fine-tunes a small transformer
(`distilbert-base-uncased`) on Yelp reviews to flag **fake vs real** reviews,
then explains each decision with **LIME**, **Integrated Gradients**, and a
**Reason Code** taxonomy — surfaced in an interactive web UI that also includes a
**Human-in-the-Loop (HITL)** evaluation to measure whether XAI explanations help
people spot fake reviews.

## Stack

- **Backend:** FastAPI (single worker), serving a JSON API + the static SPA.
- **Frontend:** vanilla HTML/CSS/JS — no build step, no CDN.
- **ML:** PyTorch + HuggingFace `transformers`, `lime`, `captum` (IG), optional
  `shap` (precomputed examples only).

## Quick start

```bash
# 1. Install dependencies (into the existing .venv)
uv pip install --python .venv/bin/python -e ".[all]"

# 2. Verify the environment
.venv/bin/python -m scripts.bootstrap

# 3. One-time training (downloads ~0.3–0.6 GB data + ~250 MB model; ~15–45 min on CPU)
make train

# 4. Run the app
make run            # -> http://127.0.0.1:8000
```

## What you get

| Page | URL | What it does |
|---|---|---|
| **Analyzer** | `/` | Paste a review → Real/Fake prediction, confidence, LIME-highlighted suspicious words, Reason Codes with evidence, LIME importance bars, Integrated-Gradients token coloring. |
| **HITL** | `/hitl.html` | Two-pass evaluation (raw text, then the same reviews WITH XAI) recording judgments, confidence and feedback; results show human accuracy/precision/recall/F1 and Cohen's κ vs the model, per mode. |
| **Examples** | `/examples.html` | Precomputed TP/TN/FP/FN reviews with LIME, IG and (if built) SHAP. |

## API

- `GET /api/health` — model readiness.
- `POST /api/analyze` `{text}` → prediction, confidence, `lime_features`,
  `ig_tokens`/`ig_attrs`, `highlighted_html`, `reason_codes`, `summary`, `elapsed_ms`.
- `GET /api/hitl/session` / `POST /api/hitl/judgment` / `GET /api/hitl/results`
  / `POST /api/hitl/reset` — the two-pass HITL flow.
- `GET /api/examples` — precomputed example explanations.

## Training

`python -m training.train` (or `make train`). Options:

```bash
.venv/bin/python -m training.train \
  --sample 20000 --epochs 3 --max_len 256 --device cpu \
  [--with-shap]   # precompute SHAP for the Examples page (slow on CPU)
```

Artifacts: `data/splits.csv`, `models/fake_review_distilbert/`
(model + tokenizer), `models/label_map.json`, `models/training_meta.json`,
`data/hitl_pool.json` (20 balanced samples with precomputed explanations),
`data/examples.json` (TP/TN/FP/FN + LIME/IG/optional SHAP).

Smoke-test the whole pipeline on a tiny subset: `make smoke`.

## How the port maps to the notebook

- **Phase 1 (data/preprocessing):** `training/data.py` + `app/preprocess.py`.
  The canonical `preprocess()` = `clean_text()` → `remove_stopwords()` is applied
  identically at train *and* serve time (the notebook does the same in cells
  10+12→16 and cell 36).
- **Phase 2 (models):** `training/train.py` — plain-PyTorch fine-tune loop with
  AdamW + linear warmup, best-val-F1 checkpoint (the notebook used RoBERTa; we
  use distilbert-base for CPU-friendly training).
- **Phase 3 (XAI):** `app/services/xai_service.py` (LIME + IG), `app/reason_codes.py`
  (RC-01…RC-05 engine). SHAP is precomputed offline only — it's too slow to run
  per request on CPU.
- **Phase 4 (dashboard + HITL):** `app/routers/` + `frontend/`. A bug in the
  notebook (cell 33 never appends its generated explanations, leaving its CSV
  empty) is not reproduced.

## What "fake" means here

Faithful to the notebook, the training label is a **star-rating proxy**: Yelp
1–2★ reviews are labelled `fake` (1) and 4–5★ `real` (0), 3★ dropped. So the
model is really detecting *low-star / harsh* reviews, and a praise-heavy
promotional review ("best ever! five stars!") is usually predicted REAL — even
though the rule-based Reason Codes (e.g. RC-01 promotional language) will still
flag its promotional wording. That tension is inherent to the notebook's
methodology and is reproduced as-is.

## Notes / tuning

- **Live XAI speed:** `/api/analyze` targets ~3–8 s on CPU. To go faster, reduce
  `LIME_NUM_SAMPLES` / `IG_N_STEPS` in `app/config.py`, or lower `MAX_LENGTH`.
- **Single worker is intentional** — the model is loaded once per process;
  a `threading.Lock` serializes expensive XAI work.
- **SHAP:** if `shap`/`numba` won't install on your Python version, the app and
  training still work; the Examples page just shows LIME/IG with a note.
