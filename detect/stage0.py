"""Stage 0 — zero-shot baseline for machine-generated review detection.

Answers three questions before any training happens:

1. **Does the signal exist at all?** Compute the `detect.features` zero-shot
   signals on real Yelp reviews vs. text from local generators, and measure
   per-feature AUC. Two generators bracket the truth: one matched to the
   reference LM (GPT-2, circular — an optimistic upper bound) and one from a
   different family (Pythia, a pessimistic lower bound). A modern API LLM sits
   between them.

2. **How much does the current preprocessing destroy?** `app.preprocess`
   lowercases, strips punctuation/casing/digits and removes stopwords. Those
   are the signals. Re-running the same AUC on preprocessed text quantifies the
   loss — this is the evidence for the raw-text serving path.

3. **Is the shipped model actually doing this task?** Correlate the trained
   `fake_review_distilbert` P(fake) against star rating and against the
   zero-shot machine-ness features on *human* reviews. If it tracks stars and
   not curvature, it is a sentiment classifier wearing a fake-review label.

Usage:
    python -m detect.stage0 [--n-human 2000] [--n-machine 300] [--skip-model]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.preprocess import normalize_escapes, preprocess
from training import config as tc

from .features import SurprisalExtractor, build_feature_row

REPORT_PATH = tc.DATA_DIR / "stage0_report.json"
FEATURES_PATH = tc.DATA_DIR / "stage0_features.csv"

# Bookkeeping columns, not discriminative signals.
_NON_SIGNAL = {"surp_n_tokens", "style_n_words"}

# (model id, tag). The tags record how far each generator sits from the
# reference LM: `matched` is circular, `diff` is a genuine cross-family test.
_GENERATORS = [
    ("gpt2", "gpt2_matched"),
    ("EleutherAI/pythia-160m", "pythia_diff"),
]

_PROMPT_TEMPLATES = [
    "I went to {name} last week and the food was ",
    "{name} is a restaurant in town. My experience there: ",
    "We had dinner at {name} on Saturday. ",
    "Review of {name}: ",
]
_NAMES = [
    "this place", "a small Italian bistro", "the new ramen shop",
    "a seafood grill downtown", "a family diner", "the sushi bar", "a taco truck",
]


def load_yelp_raw(n: int, seed: int = 7) -> pd.DataFrame:
    """Raw (unpreprocessed) Yelp reviews with an explicit 1-5 star rating.

    The HF dataset encodes `label` as 0-4 (= 1-5 stars); we shift to real star
    counts so nothing downstream has to remember that offset.

    All five levels are kept. For machine-text detection the rating is not a
    label at all — it is a *controlled variable* we must hold balanced so the
    detector cannot shortcut on sentiment the way the shipped model does.
    """
    import datasets

    ds = datasets.load_dataset(
        "Yelp/yelp_review_full", split="train", cache_dir=str(tc.HF_CACHE_DIR)
    )
    df = ds.to_pandas()
    df["stars"] = df["label"] + 1  # 0-4 -> 1-5
    # Undo the corpus's escape artefact before anything measures the text.
    df["text"] = df["text"].map(normalize_escapes)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df[["text", "stars"]]


def generate_texts(model_name: str, n: int, seed: int, device: str,
                   max_new_tokens: int = 140) -> list[str]:
    """Sample `n` short restaurant-review-shaped texts from a local LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(tc.HF_CACHE_DIR))
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(tc.HF_CACHE_DIR))
    model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rng = np.random.RandomState(seed)
    prompts = [
        _PROMPT_TEMPLATES[i % len(_PROMPT_TEMPLATES)].format(
            name=_NAMES[rng.randint(len(_NAMES))]
        )
        for i in range(n)
    ]

    out: list[str] = []
    batch = 16
    with torch.no_grad():
        for i in range(0, n, batch):
            chunk = prompts[i : i + batch]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            for row, prompt in zip(gen, chunk):
                text = tokenizer.decode(row, skip_special_tokens=True)
                text = text[len(prompt):] if text.startswith(prompt) else text
                out.append(text.strip())

    del model
    if device == "mps":
        torch.mps.empty_cache()
    # Drop empties / fragments too short to carry any style signal.
    return [t for t in out if len(t.split()) >= 15]


def _extract_rows(texts: list[str], extractor: SurprisalExtractor,
                  batch_size: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        for text, surp in zip(chunk, extractor.extract(chunk)):
            rows.append(build_feature_row(surp, text))
    return pd.DataFrame(rows)


def _auc_table(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """AUC of each feature at separating machine (label=1) from human (0)."""
    y = df["is_machine"].to_numpy()
    out = {}
    for col in feature_cols:
        x = df[col].to_numpy(dtype=float)
        if not np.isfinite(x).all() or np.nanstd(x) == 0:
            continue
        out[col] = float(roc_auc_score(y, x))
    return out


def _ranked(aucs: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Top-k features by separation strength, direction-normalised to >=0.5."""
    return sorted(aucs.items(), key=lambda kv: -max(kv[1], 1 - kv[1]))[:k]


def _fmt(name: str, auc: float) -> str:
    """Show the separation strength and flag direction flips."""
    strength = max(auc, 1 - auc)
    arrow = "↓machine" if auc < 0.5 else "↑machine"
    return f"{name}={strength:.3f}{arrow}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 0 zero-shot baseline.")
    parser.add_argument("--n-human", type=int, default=2000)
    parser.add_argument("--n-machine", type=int, default=300)
    parser.add_argument("--ablation-n", type=int, default=200,
                        help="rows per side for the preprocessing ablation")
    parser.add_argument("--device", default="mps" if _mps() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-model", action="store_true",
                        help="skip the shipped-model correlation analysis")
    args = parser.parse_args(argv)
    t0 = time.time()

    tc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_human": args.n_human,
        "n_machine": args.n_machine,
        "reference_lm": "gpt2",
    }

    print(f"== Loading {args.n_human} raw Yelp reviews ==")
    human = load_yelp_raw(args.n_human, args.seed)
    print(f"   {len(human)} unique reviews; star counts "
          f"{human['stars'].value_counts().sort_index().to_dict()}")

    print("== Extracting zero-shot features from human reviews (GPT-2) ==")
    extractor = SurprisalExtractor(cache_dir=str(tc.HF_CACHE_DIR), device=args.device)
    human_feats = _extract_rows(human["text"].tolist(), extractor)
    human_feats["stars"] = human["stars"].to_numpy()
    human_feats["is_machine"] = 0
    human_feats["source"] = "yelp_human"

    feature_cols = [c for c in human_feats.columns
                    if c.startswith(("surp_", "style_")) and c not in _NON_SIGNAL]

    # ------------------------------------------------------- 1. signal check
    print(f"== Generating {args.n_machine} texts per local LM ==")
    generated: dict[str, list[str]] = {}
    machine_frames = []
    for model_id, tag in _GENERATORS:
        texts = generate_texts(model_id, args.n_machine, args.seed, args.device)
        if len(texts) < 30:
            print(f"   {tag}: only {len(texts)} usable samples, skipped")
            continue
        generated[tag] = texts
        feats = _extract_rows(texts, extractor)
        feats["stars"] = -1
        feats["is_machine"] = 1
        feats["source"] = tag
        machine_frames.append(feats)
        print(f"   {tag}: {len(texts)} samples from {model_id}")

    if not machine_frames:
        raise SystemExit("No machine samples generated — cannot run the sanity check.")

    print("\n== Signal check: per-feature AUC (machine vs human) ==")
    report["signal_auc"] = {}
    for frame in machine_frames:
        tag = frame["source"].iloc[0]
        aucs = _auc_table(pd.concat([human_feats, frame], ignore_index=True), feature_cols)
        report["signal_auc"][tag] = aucs
        print(f"\n  -- {tag} (n_machine={len(frame)}, "
              f"top features by separation) --")
        for name, auc in _ranked(aucs, 8):
            print(f"     {name:26s} {_fmt(name.split('_', 1)[1], auc)}")

    # --------------------------------------------------- 2. preprocessing ablation
    # Same comparison, but with the shipped preprocessing applied first. If the
    # best achievable separation collapses, the raw-text path is load-bearing.
    ab_n = min(args.ablation_n, len(generated.get("pythia_diff", [])), len(human))
    if ab_n >= 30:
        print(f"\n== Preprocessing ablation (n={ab_n} per side, pythia_diff) ==")
        h_raw = human["text"].tolist()[:ab_n]
        m_raw = generated["pythia_diff"][:ab_n]
        report["preprocessing_ablation"] = {}
        for label, transform in [("raw", lambda s: s), ("preprocessed", preprocess)]:
            h = _extract_rows([transform(t) for t in h_raw], extractor)
            m = _extract_rows([transform(t) for t in m_raw], extractor)
            h["is_machine"], m["is_machine"] = 0, 1
            aucs = _auc_table(pd.concat([h, m], ignore_index=True), feature_cols)
            report["preprocessing_ablation"][label] = aucs
            best = max(aucs.items(), key=lambda kv: max(kv[1], 1 - kv[1]))
            print(f"   {label:13s} best={_fmt(best[0].split('_', 1)[1], best[1])}  |  "
                  + "  ".join(_fmt(k.split('_', 1)[1], v) for k, v in _ranked(aucs, 4)))
        best_auc = {
            label: max(max(a.values(), default=0.5), 1 - min(a.values(), default=0.5))
            for label, a in report["preprocessing_ablation"].items()
        }
        report["preprocessing_ablation_best"] = best_auc
        print(f"   -> best separation drops {best_auc['raw']:.3f} (raw) "
              f"to {best_auc['preprocessed']:.3f} (preprocessed)")

    # ------------------------------------------------------ 3. shipped model
    if args.skip_model:
        pass
    elif not tc.OUTPUT_DIR.exists():
        print("\n== Shipped model not found; skipping correlation analysis ==")
    else:
        print("\n== Shipped model vs. the real task (HUMAN reviews only) ==")
        probs = _shipped_model_probs(human["text"].tolist(), args.device)
        if probs is not None:
            human_feats["shipped_p_fake"] = probs
            stars = human_feats["stars"].to_numpy(dtype=float)
            block: dict[str, float] = {
                # Reconstructs the shipped model's own training target
                # (1-2 stars = fake) on a fresh human sample. A high value here
                # means it is solving the sentiment proxy, not machine-text
                # detection — cross-check against the curvature correlation.
                "auc_pfake_vs_star_le2": float(
                    roc_auc_score((stars <= 2).astype(int), probs)
                ),
                "pearson_pfake_stars": float(np.corrcoef(probs, stars)[0, 1]),
            }
            for col in ["surp_curvature", "surp_lp_mean", "surp_ppl", "style_sent_len_cv"]:
                x = human_feats[col].to_numpy(dtype=float)
                if np.isfinite(x).all() and np.nanstd(x) > 0:
                    block[f"pearson_pfake_{col}"] = float(np.corrcoef(probs, x)[0, 1])
            curv = human_feats["surp_curvature"].to_numpy(dtype=float)
            block["pearson_curvature_stars"] = float(np.corrcoef(curv, stars)[0, 1])
            report["shipped_model"] = block

            print(f"   P(fake) vs 1-2 star ............. AUC "
                  f"{block['auc_pfake_vs_star_le2']:.3f}   (its own training target)")
            print(f"   corr(P(fake), stars) ............ "
                  f"{block['pearson_pfake_stars']:+.3f}")
            print(f"   corr(P(fake), curvature) ........ "
                  f"{block.get('pearson_pfake_surp_curvature', float('nan')):+.3f}")
            print(f"   corr(curvature, stars) .......... "
                  f"{block['pearson_curvature_stars']:+.3f}")

    # ---------------------------------------------------------------- output
    human_feats.to_csv(FEATURES_PATH, index=False)
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFeatures -> {FEATURES_PATH}\nReport   -> {REPORT_PATH}")
    print(f"== Stage 0 complete in {report['elapsed_seconds']}s ==")


def _shipped_model_probs(texts: list[str], device: str):
    """P(fake) from the shipped checkpoint, on the text it was trained for."""
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(str(tc.OUTPUT_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(tc.OUTPUT_DIR))
        model.to(device).eval()
        probs = []
        clean = [preprocess(t) for t in texts]
        with torch.no_grad():
            for i in range(0, len(clean), 16):
                enc = tokenizer(clean[i : i + 16], truncation=True, padding="max_length",
                                max_length=tc.MAX_LENGTH, return_tensors="pt").to(device)
                logits = model(**enc).logits
                probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        del model
        return np.concatenate(probs)
    except Exception as exc:  # noqa: BLE001
        print(f"   [shipped model unavailable: {type(exc).__name__}: {exc}]")
        return None


def _mps() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
