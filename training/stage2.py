"""Stage 2 — train and evaluate the machine-generated-review detector.

Trains on the zero-shot signals in `detect.features` (reference-LM surprisal +
stylometric), computed on **raw** text. `app.preprocess.preprocess` is never
applied: Stage 0 measured that it destroys ~0.19 AUC of the available signal by
lowercasing, stripping punctuation and removing stopwords.

Reporting rules, chosen so the eventual second generator is a new row and not a
rewrite of the analysis:

- **Per-generator rows always.** With one generator the table has one row; the
  code path does not change when there are two.
- **TPR at a fixed FPR, not F1.** In production a false positive means accusing
  a real customer, so the operating point is chosen by the false-positive rate
  and the headline number is how much machine text is caught there.
- **Cross-validated error bars**, because the frozen dev split holds only ~150
  machine records and a single number off that would be noise.
- **Probe false-positive rates** on raw held-out Yelp and on copy-edited human
  reviews. The second is the one that catches "learned that clean text means
  machine".

Usage:
    python -m training.stage2 [--generators mimo-v2.5-rerun] [--no-cache]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from detect import dataset as ds
from detect.features import SurprisalExtractor, build_feature_row
from training import config as tc

CACHE_DIR = tc.DATA_DIR / "stage2_cache"
REPORT_PATH = tc.DATA_DIR / "stage2_report.json"
DETECTOR_PATH = tc.MODEL_DIR / "stage2_detector.joblib"
REFERENCE_PATH = tc.MODEL_DIR / "stage2_reference.json"

FPR_TARGETS = (0.01, 0.05)
N_FOLDS = 5


# ---------------------------------------------------------------- features

def extract_features(records, cache_dir: Path, name: str, use_cache: bool,
                     device: str) -> pd.DataFrame:
    """Feature matrix for `records`, cached per split on disk.

    Each split gets its own file (the train matrix would otherwise be evicted by
    the next probe), and the cache is keyed on record ids so rebuilding the
    dataset — a new generator, a changed packet — invalidates it rather than
    silently reusing rows that no longer line up.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.csv"
    key = "|".join(r.record_id for r in records)
    if use_cache and cache.exists():
        cached = pd.read_csv(cache)
        if "__key__" in cached.columns and cached["__key__"].iloc[0] == key:
            print(f"[features] {name}: cache hit ({len(cached)} rows)")
            return cached.drop(columns=["__key__"])

    print(f"[features] {name}: extracting {len(records)} records")
    extractor = SurprisalExtractor(cache_dir=str(tc.HF_CACHE_DIR), device=device)
    rows, t0 = [], time.time()
    for i in range(0, len(records), 4):
        chunk = records[i : i + 4]
        for rec, surp in zip(chunk, extractor.extract([r.text for r in chunk])):
            row = build_feature_row(surp, rec.text)
            row["__key__"] = key
            rows.append(row)
        if (i // 4) % 50 == 0:
            print(f"  {i + len(chunk)}/{len(records)}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df.drop(columns=["__key__"])


# Bookkeeping, not style. `n_words`/`n_tokens` are excluded because the
# generator prompt pins output to 60-160 words while real Yelp reviews run
# 32-371, so a bare length count partly encodes *our instruction*, not machine
# authorship. They stay in the frame for the length-matched check, which needs
# them to define the window.
BOOKKEEPING = ("style_n_words", "surp_n_tokens")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith(("surp_", "style_")) and c not in BOOKKEEPING]


def length_matched_mask(X: pd.DataFrame, y: np.ndarray,
                        lo_pct: float = 5, hi_pct: float = 95) -> np.ndarray:
    """Rows whose word count falls inside the machine class's central range.

    The generator's length instruction makes machine output cluster tightly; a
    model can exploit that gap without learning anything about authorship. This
    restricts both classes to the window where they actually overlap, which is
    the honest way to ask whether the signal survives without the shortcut.
    """
    w = X["style_n_words"].to_numpy(dtype=float)
    machine = w[y == 1]
    lo, hi = np.percentile(machine, lo_pct), np.percentile(machine, hi_pct)
    return (w >= lo) & (w <= hi)


# ---------------------------------------------------------------- metrics

def tpr_at_fpr(y_true, scores, target: float) -> float:
    """Highest TPR achievable while FPR stays at or below `target`."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(y_true, scores, target: float) -> float:
    """Lowest threshold whose FPR is still within `target`."""
    fpr, _, thr = roc_curve(y_true, scores)
    ok = np.where(fpr <= target)[0]
    return float(thr[ok[np.argmax(fpr[ok])]]) if len(ok) else 1.0


def evaluate(y_true, scores) -> dict:
    out: dict = {"n": int(len(y_true)), "n_pos": int(np.sum(y_true))}
    if len(set(np.asarray(y_true))) < 2:
        return out
    out["auroc"] = round(float(roc_auc_score(y_true, scores)), 4)
    for t in FPR_TARGETS:
        out[f"tpr_at_fpr_{int(t * 100)}pct"] = round(tpr_at_fpr(y_true, scores, t), 4)
    return out


def _fpr(records, scores, threshold: float) -> dict:
    flagged = int(np.sum(np.asarray(scores) >= threshold))
    return {"n": len(records), "flagged": flagged,
            "fpr": round(flagged / max(1, len(records)), 4)}


# ---------------------------------------------------------------- groups

def per_group(model, cols, X, records) -> dict:
    """Metrics split by generator and by condition.

    This is the table that grows a row when a second generator is added — the
    cross-generator number is the difference between the rows, so it must never
    be collapsed into one pooled figure.
    """
    scores = model.predict_proba(X[cols])[:, 1]
    y = np.array([r.label for r in records])
    human_idx = np.array([i for i, r in enumerate(records) if r.label == 0])

    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        if r.label == 1 and r.generator:
            groups.setdefault(r.generator, []).append(i)
    for cond in sorted({r.condition for r in records if r.label == 1}):
        groups[f"by_condition::{cond}"] = [i for i, r in enumerate(records)
                                           if r.condition == cond]

    # Every machine subset is scored against the same human baseline, so the
    # rows stay directly comparable — and the difference between two
    # generators' rows is the cross-generator generalisation gap.
    out = {}
    for name, idx in sorted(groups.items()):
        sel = np.concatenate([np.array(idx), human_idx])
        out[name] = evaluate(y[sel], scores[sel]) | {
            "machine_n": int(np.sum(y[np.array(idx)] == 1))
        }
    out["human_baseline_only"] = {
        "n": int(len(human_idx)),
        "mean_score": round(float(scores[human_idx].mean()), 4),
    }
    return out


def ablation(model, cols, Xtr, ytr, Xte, yte, Xpr, probes, seed: int) -> dict:
    """Which feature families actually carry the decision.

    Reported alongside permutation importance because the two disagree in a
    useful way: `style_avg_word_len` tops the permutation table but removing it
    changes nothing, so it is redundant rather than load-bearing. Only the
    refit tells you that.
    """
    families = {
        "all": cols,
        "surprisal_only": [c for c in cols if c.startswith("surp_")],
        "stylometric_only": [c for c in cols if c.startswith("style_")],
        "without_avg_word_len": [c for c in cols if c != "style_avg_word_len"],
        "single_lp_mean": ["surp_lp_mean"],
        "single_avg_word_len": ["style_avg_word_len"],
    }
    out = {}
    for name, sel in families.items():
        if not sel:
            continue
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(ytr))
        for tr_idx, va_idx in cv.split(Xtr[sel], ytr):
            m = clone(model).fit(Xtr[sel].iloc[tr_idx], ytr[tr_idx])
            oof[va_idx] = m.predict_proba(Xtr[sel].iloc[va_idx])[:, 1]
        fitted = clone(model).fit(Xtr[sel], ytr)
        thr = threshold_at_fpr(ytr, oof, 0.01)
        out[name] = {
            "n_features": len(sel),
            "cv_auroc": round(float(roc_auc_score(ytr, oof)), 4),
            "test_auroc": round(float(roc_auc_score(yte, fitted.predict_proba(Xte[sel])[:, 1])), 4),
            "probe_fpr": {
                k: _fpr(probes[k], fitted.predict_proba(Xpr[k][sel])[:, 1], thr)["fpr"]
                for k in Xpr
            },
        }
    return out


def feature_importance(model, X, y, cols) -> list[dict]:
    """Permutation importance on held-out folds — model-agnostic and honest."""
    from sklearn.inspection import permutation_importance

    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    tr, va = next(iter(cv.split(X, y)))
    m = clone(model).fit(X.iloc[tr], y[tr])
    r = permutation_importance(m, X.iloc[va], y[va], n_repeats=10, random_state=0,
                               scoring="roc_auc")
    order = np.argsort(-r.importances_mean)
    return [
        {"feature": cols[i], "auc_drop": round(float(r.importances_mean[i]), 4),
         "std": round(float(r.importances_std[i]), 4)}
        for i in order[:12]
    ]


# ---------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the Stage 2 detector.")
    # Defaults to the only run verified to be genuine model output. Do NOT point
    # this at mimo-v2.5-full: that run came from scripts/gen_reviews.py, and
    # training on it separates template text from human text rather than machine
    # text from human text.
    p.add_argument("--generators", nargs="+", default=["mimo-v2.5-rerun"],
                   help="Stage 1 run tags forming the machine class")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--device", default="mps" if _mps() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    t0 = time.time()
    tc.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, test, probes = ds.build(args.generators)
    print("== Stage 2 dataset ==")
    for name, part in [("train", train), ("test", test)]:
        s = ds.summarize(part)
        print(f"  {name:5s} n={s['n']:5d}  label={s['label']}  cond={s['condition']}")
    for name, part in probes.items():
        print(f"  probe {name:20s} n={len(part):5d}")

    print("\n== Features (raw text) ==")
    use_cache = not args.no_cache
    Xtr = extract_features(train, CACHE_DIR, "train", use_cache, args.device)
    Xte = extract_features(test, CACHE_DIR, "test", use_cache, args.device)
    Xpr = {k: extract_features(v, CACHE_DIR, k, use_cache, args.device)
           for k, v in probes.items() if v}
    cols = feature_columns(Xtr)
    ytr = np.array([r.label for r in train])
    yte = np.array([r.label for r in test])
    print(f"  {len(cols)} features | {len(Xtr)} train / {len(Xte)} test")

    report: dict = {
        "generators": args.generators,
        "n_features": len(cols),
        "train": ds.summarize(train),
        "test": ds.summarize(test),
    }

    # ---- zero-shot baseline: single strongest raw feature ------------------
    aucs = {c: roc_auc_score(ytr, Xtr[c]) for c in cols}
    best = max(aucs, key=lambda c: max(aucs[c], 1 - aucs[c]))
    sign = 1.0 if aucs[best] >= 0.5 else -1.0
    report["baseline_single_feature"] = {
        "feature": best, "sign": sign,
        "train_auroc": round(max(aucs[best], 1 - aucs[best]), 4),
        "test": evaluate(yte, sign * Xte[best].to_numpy()),
    }
    print(f"\n== Baseline: single zero-shot feature '{best}' "
          f"(train AUROC {report['baseline_single_feature']['train_auroc']}) ==")
    print(f"  test {report['baseline_single_feature']['test']}")

    # ---- models -----------------------------------------------------------
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=args.seed),
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced",
            random_state=args.seed,
        ),
    }

    report["models"] = {}
    for name, model in models.items():
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=args.seed)
        cv_aucs, oof = [], np.zeros(len(ytr))
        for tr_idx, va_idx in cv.split(Xtr[cols], ytr):
            m = clone(model).fit(Xtr[cols].iloc[tr_idx], ytr[tr_idx])
            oof[va_idx] = m.predict_proba(Xtr[cols].iloc[va_idx])[:, 1]
            cv_aucs.append(roc_auc_score(ytr[va_idx], oof[va_idx]))

        model.fit(Xtr[cols], ytr)
        block = {
            "cv_auroc_mean": round(float(np.mean(cv_aucs)), 4),
            "cv_auroc_std": round(float(np.std(cv_aucs)), 4),
            "cv_oof": evaluate(ytr, oof),
            "test": evaluate(yte, model.predict_proba(Xte[cols])[:, 1]),
        }
        # Operating point is fixed on out-of-fold training scores, never on the
        # test split — otherwise the probes inherit an optimistically placed
        # threshold.
        thr = threshold_at_fpr(ytr, oof, 0.01)
        block["threshold_at_train_fpr_1pct"] = round(thr, 5)
        block["probes"] = {
            k: _fpr(probes[k], model.predict_proba(Xpr[k][cols])[:, 1], thr)
            for k in Xpr
        }
        block["per_group"] = per_group(model, cols, Xte, test)

        # Robustness: refit on the length-overlap window only.
        mask = length_matched_mask(Xtr, ytr)
        m_cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=args.seed)
        oof_m = np.zeros(int(mask.sum()))
        Xm, ym = Xtr[cols][mask], ytr[mask]
        for tr_idx, va_idx in m_cv.split(Xm, ym):
            m = clone(model).fit(Xm.iloc[tr_idx], ym[tr_idx])
            oof_m[va_idx] = m.predict_proba(Xm.iloc[va_idx])[:, 1]
        block["length_matched"] = {
            "n": int(mask.sum()),
            "n_pos": int(ym.sum()),
            "cv_auroc": round(float(roc_auc_score(ym, oof_m)), 4),
        }
        report["models"][name] = block

        print(f"\n== {name} ==")
        print(f"  CV AUROC {block['cv_auroc_mean']:.4f} +/- {block['cv_auroc_std']:.4f}"
              f"   (out-of-fold, n={len(ytr)})")
        print(f"  test     {block['test']}")
        print(f"  length-matched {block['length_matched']}")
        print(f"  probes   {block['probes']}")

    # Single-feature AUC first: with 24 correlated features, permutation
    # importance on one held-out fold understates every one of them, and the
    # univariate table is the more honest picture of what carries signal.
    report["single_feature_auroc"] = sorted(
        ({"feature": c, "auroc": round(float(roc_auc_score(ytr, Xtr[c])), 4)}
         for c in cols),
        key=lambda r: -abs(r["auroc"] - 0.5),
    )
    print("\n== Single-feature AUROC (train) ==")
    for row in report["single_feature_auroc"][:10]:
        print(f"  {row['feature']:26s} {row['auroc']:.3f}")

    report["feature_importance"] = feature_importance(models["hgb"], Xtr[cols], ytr, cols)
    print("\n== Permutation importance (held-out fold) ==")
    for row in report["feature_importance"][:6]:
        print(f"  {row['feature']:26s} dAUC {row['auc_drop']:+.4f} +/- {row['std']:.4f}")

    report["ablation"] = ablation(models["hgb"], cols, Xtr, ytr, Xte, yte, Xpr, probes,
                                  args.seed)
    print("\n== Ablation (hgb refit per family) ==")
    for name, b in report["ablation"].items():
        print(f"  {name:22s} n_feat={b['n_features']:2d}  CV {b['cv_auroc']:.4f}"
              f"  test {b['test_auroc']:.4f}  probes {b['probe_fpr']}")

    print("\n== Per-generator / per-condition (frozen dev split) ==")
    for name in models:
        for group, m in report["models"][name]["per_group"].items():
            if "auroc" not in m:
                continue
            print(f"  {name:7s} {group:30s} AUROC {m['auroc']:.4f}"
                  f"  TPR@1%FPR {m.get('tpr_at_fpr_1pct', float('nan')):.3f}"
                  f"  (n_pos={m['n_pos']})")

    save_artifacts(
        models["hgb"], cols, Xtr[cols], ytr,
        report["models"]["hgb"]["threshold_at_train_fpr_1pct"],
        args.generators, ("gpt2_surprisal", "stylometric_raw_text"),
    )

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {REPORT_PATH}")


def save_artifacts(model, cols, Xtr, ytr, threshold: float, generators: list[str],
                   feats: tuple[str, ...]) -> None:
    """Persist the fitted detector plus the human/machine reference profile.

    The serving layer needs the profile, not just the model: a reason code is
    only meaningful as a position within the human distribution ("this review's
    mean surprisal sits at the 3rd percentile of real reviews"), and the
    deployment text is scored with exactly the feature set and column order the
    model was fitted on.
    """
    import joblib

    joblib.dump(
        {"model": model, "features": list(cols), "bookkeeping": list(BOOKKEEPING),
         "threshold_at_fpr_1pct": threshold, "generators": generators},
        DETECTOR_PATH,
    )

    profile = {}
    for c in cols:
        h = Xtr[c][ytr == 0].to_numpy(dtype=float)
        m = Xtr[c][ytr == 1].to_numpy(dtype=float)
        profile[c] = {
            "human_pct": [round(float(np.percentile(h, p)), 6)
                          for p in (1, 5, 25, 50, 75, 95, 99)],
            "machine_pct": [round(float(np.percentile(m, p)), 6)
                            for p in (1, 5, 25, 50, 75, 95, 99)],
        }
    with open(REFERENCE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "features": list(cols),
            "feature_extractor": feats,
            "generators": generators,
            "profile": profile,
        }, f, indent=2)
    print(f"Detector -> {DETECTOR_PATH}\nReference -> {REFERENCE_PATH}")


def _mps() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
