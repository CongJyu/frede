"""Score genuine machine text from other models with the trained detector.

This is the cross-generator evaluation, and it needs no second generation run:
**evaluation requires only the text, not alignment to the packet.** Genuine
output from other models is already on disk, from runs rejected for *training*
because their custom_ids pointed at the wrong reviews. A record that cannot be
trained on can still be scored.

The question it answers: does the detector read authorship, or one generator's
house style? Score it on text from models it never saw and watch the decay.

One trap worth recording. Scoring GPT-2 or Pythia output gives a median
P(machine) near zero, which looks like total failure to transfer. It is not
informative: those are base language models whose continuation does not resemble
a *review* — and the task is detecting model-written reviews, not any LM output.
A generator only probes the right thing if it writes actual reviews.

Usage:
    python -m scripts.cross_generator [--generator mimo-v2.5-rerun] [--n 400]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detect.dataset import build
from detect.features import build_feature_row
from training import config as tc

RESULTS_DIR = tc.DATA_DIR / "gen_results"

# (run directory, condition, label). The first entry is the detector's own
# training data and serves as the in-distribution reference point.
SOURCES = [
    ("mimo-v2.5-rerun", "machine_generate", "MiMo rerun — TRAINED ON THIS"),
    ("mimo-v2.5-opencode", "machine_generate", "MiMo first run — same model, new session"),
    ("deepseek-v4f-cline", "machine_generate", "DeepSeek V4F — different model"),
    ("deepseek-v4f-cline", "machine_rewrite", "DeepSeek V4F rewrites"),
    ("mimo-v2.5-opencode", "machine_rewrite", "MiMo first-run rewrites"),
]


def load_texts(run: str, condition: str, limit: int) -> list[str]:
    out: list[str] = []
    for path in sorted((RESULTS_DIR / run).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("custom_id", "")).endswith(condition):
                out.append(rec["text"])
            if len(out) >= limit:
                return out
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Cross-generator evaluation.")
    p.add_argument("--generator", default="mimo-v2.5-rerun",
                   help="the run the detector was trained on")
    p.add_argument("--n", type=int, default=400, help="records sampled per source")
    args = p.parse_args(argv)

    from app.services.detector_service import get_detector_service

    ds = get_detector_service()
    ds.warmup()
    print(f"detector trained on {ds.generators}, threshold {ds.threshold:.4f}\n")

    def score(texts: list[str]) -> np.ndarray:
        rows = []
        for i in range(0, len(texts), 4):
            chunk = texts[i : i + 4]
            for text, surp in zip(chunk, ds.extractor.extract(chunk)):
                rows.append(build_feature_row(surp, text))
        X = pd.DataFrame([{c: r.get(c, np.nan) for c in ds.features} for r in rows])
        return ds.model.predict_proba(X[ds.features])[:, 1]

    print(f"{'source':44s} {'n':>4s} {'median P':>9s} {'flagged':>8s}")
    for run, condition, label in SOURCES:
        texts = load_texts(run, condition, args.n)
        if len(texts) < 20:
            print(f"{label:44s} {len(texts):4d}   (too few records)")
            continue
        probs = score(texts)
        print(f"{label:44s} {len(texts):4d} {np.median(probs):9.3f} "
              f"{(probs >= ds.threshold).mean():7.1%}")

    _, test, _ = build([args.generator])
    for label, texts in [
        ("human, held-out Yelp (reference)", [r.text for r in test if r.label == 0]),
        (f"{args.generator}, in-distribution test", [r.text for r in test if r.label == 1]),
    ]:
        probs = score(texts)
        print(f"{label:44s} {len(texts):4d} {np.median(probs):9.3f} "
              f"{(probs >= ds.threshold).mean():7.1%}")

    print(
        "\nReading: same model in a new session should barely move; a different "
        "model\nshould drop but stay far from human. Either extreme is a finding "
        "— no\nmovement means a house-style detector, no drop means a general one."
    )


if __name__ == "__main__":
    main()
