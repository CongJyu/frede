"""Precompute the balanced HITL sample pool with explanations.

The pool is drawn from the detector's held-out split and split evenly between
genuinely human-written and genuinely machine-written reviews. Ground truth is
now actual authorship — real Yelp reviews on one side, verified model output on
the other — rather than the star-rating proxy the previous study used. That
makes "human accuracy" mean what it says.

Explanations are computed once here with the same `run_analysis` the Analyzer
page calls, so what a participant sees during the study is exactly what the live
tool shows. The previous version reimplemented the explanation stack offline
with LIME and Integrated Gradients; that existed only because the old model
needed sampling-based attribution. This detector's explanations are exact and
cheap, so there is nothing to reimplement.

Usage:
    python -m training.hitl_pool [--generator mimo-v2.5-rerun] [--n 20]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from detect.dataset import build
from training import config as tc


def build_hitl_pool(generator: str, n: int, seed: int) -> None:
    from app.services.detector_service import get_detector_service
    from app.services.xai_service import run_analysis

    ds = get_detector_service()
    ds.warmup()

    _, test, _ = build([generator])
    machine = [r for r in test if r.label == 1]
    human = [r for r in test if r.label == 0]
    if not machine or not human:
        raise SystemExit(f"Test split has {len(machine)} machine / {len(human)} human records.")

    rng = np.random.RandomState(seed)
    n_each = n // 2
    picked = list(rng.choice(machine, size=min(n_each, len(machine)), replace=False)) + \
        list(rng.choice(human, size=min(n_each, len(human)), replace=False))
    rng.shuffle(picked)

    samples = []
    for i, rec in enumerate(picked):
        r = run_analysis(rec.text)
        samples.append({
            "idx": i,
            "record_id": rec.record_id,
            "text": rec.text,
            # Ground truth is authorship, and the model's own call is stored
            # alongside it so the results page can compare the two.
            "true_label": rec.label,
            "model_pred": r["predicted"],
            "machine_prob": r["machine_prob"],
            "condition": rec.condition,
            # Explanation payload — identical in shape to the Analyzer's, so the
            # study shows participants the real tool rather than a lookalike.
            "highlighted_html": r["highlighted_html"],
            "reason_codes": r["reason_codes"],
            "summary": r["summary"],
        })
        print(f"  sample {i + 1}/{len(picked)}: "
              f"true={'machine' if rec.label else 'human'} "
              f"pred={'machine' if r['predicted'] else 'human'} "
              f"P={r['machine_prob']:.3f}")

    tc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(tc.HITL_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generator": generator,
            "n": len(samples),
            "n_machine": sum(1 for s in samples if s["true_label"] == 1),
            "n_human": sum(1 for s in samples if s["true_label"] == 0),
            "samples": samples,
        }, f, ensure_ascii=False, indent=2)
    print(f"HITL pool saved: {len(samples)} samples -> {tc.HITL_POOL_PATH}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the HITL sample pool.")
    p.add_argument("--generator", default="mimo-v2.5-rerun")
    p.add_argument("--n", type=int, default=tc.HITL_N_SAMPLES)
    p.add_argument("--seed", type=int, default=tc.HITL_SEED)
    args = p.parse_args(argv)
    build_hitl_pool(args.generator, args.n, args.seed)


if __name__ == "__main__":
    main()
