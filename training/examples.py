"""Precompute Examples-page data: one TP / TN / FP / FN case with explanations.

The four cases are the ones worth looking at — two the detector gets right and
two it gets wrong, so the page shows the failure modes rather than only the
successes. Explanations come from the same `run_analysis` the Analyzer uses, so
nothing here can drift from what the live tool reports.

Examples are chosen from the held-out split only. Picking from training data
would show the page cases the detector has already memorised.

Usage:
    python -m training.examples [--generator mimo-v2.5-rerun]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from detect.dataset import build
from training import config as tc

# Case, true label, predicted label, title, and how to pick the instance.
#
# The correct calls use the most confident instance, which shows what a clear
# signal looks like. The errors use the one nearest the decision boundary, which
# shows where the reasoning gives way — a confident miss would only say "it was
# badly wrong", while a near miss shows the actual margin. Picking the boundary
# case for every quadrant, as an earlier version did, makes all four look the
# same and teaches nothing.
CASES = [
    ("true_positive", 1, 1, "Machine-written, correctly flagged", "confident"),
    ("true_negative", 0, 0, "Human-written, correctly cleared", "confident"),
    ("false_positive", 0, 1, "Human-written, wrongly flagged", "boundary"),
    ("false_negative", 1, 0, "Machine-written, missed", "boundary"),
]


def build_examples(generator: str) -> None:
    from app.services.detector_service import get_detector_service
    from app.services.xai_service import run_analysis

    ds = get_detector_service()
    ds.warmup()

    _, test, _ = build([generator])
    scored = [(run_analysis(r.text), r) for r in test]

    examples, missing = [], []
    for key, true_label, pred_label, title, pick in CASES:
        pool = [(r, res) for res, r in scored
                if r.label == true_label and res["predicted"] == pred_label]
        if not pool:
            # A missing quadrant is a result, not a failure to fill the page: no
            # false positives means the detector flagged no real reviewer at this
            # operating point. Say so rather than quietly showing three cases.
            missing.append({"key": key, "title": title})
            print(f"  [{key}] none in the held-out split — {title}")
            continue
        if pick == "confident":
            rec, res = max(pool, key=lambda p: abs(p[1]["machine_prob"] - 0.5))
        else:
            rec, res = min(pool, key=lambda p: abs(p[1]["machine_prob"] - ds.threshold))
        examples.append({
            "key": key,
            "title": title,
            "text": rec.text,
            "condition": rec.condition,
            "true_label": true_label,
            "model_pred": pred_label,
            "machine_prob": res["machine_prob"],
            "threshold": res["threshold"],
            "highlighted_html": res["highlighted_html"],
            "reason_codes": res["reason_codes"],
            "features": res["features"][:8],
            "summary": res["summary"],
        })
        print(f"  [{key}] P(machine)={res['machine_prob']:.3f} "
              f"(threshold {res['threshold']:.3f})")

    tc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(tc.EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generator": generator,
            "threshold": round(ds.threshold, 5),
            "n_evaluated": len(test),
            "missing_cases": missing,
            "examples": examples,
        }, f, ensure_ascii=False, indent=2)
    print(f"Examples saved -> {tc.EXAMPLES_PATH}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the Examples page data.")
    p.add_argument("--generator", default="mimo-v2.5-rerun")
    args = p.parse_args(argv)
    build_examples(args.generator)


if __name__ == "__main__":
    main()
