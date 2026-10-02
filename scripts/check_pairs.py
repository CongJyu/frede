"""Health-check a generator's output before committing to a full run.

The ingest step answers "did this parse and is it plausible?". This answers the
questions that actually decide whether a run is usable, none of which are
visible from a coverage number:

- **Template collapse** — distinct outputs vs. records. A run that repeats a
  handful of reviews scores 100% coverage and is worthless; the detector would
  learn those strings and nothing else.
- **Content binding** — for `machine_rewrite`, word overlap with the source. Near
  zero means the model ignored the source and wrote something generic, which
  silently destroys the pairing the whole design rests on.
- **Sentiment preservation** — for `machine_rewrite`, correlation between source
  and output scored by the shipped classifier. Pairing is only meaningful if the
  opinion survived the rewrite.
- **Truncation** — share of outputs ending mid-sentence, plus output length
  relative to source.

Usage:
    python -m scripts.check_pairs --results data/gen_results/<tag> --generator <tag>
    python -m scripts.check_pairs --results <dir> --generator <tag> --no-sentiment
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from training import config as tc

from .ingest_gen_results import (
    PACKET_DIR,
    _ENDS_CLEAN,
    _extract_review_text,
    _jaccard,
    _validate,
    _words,
)

# A run is not trustworthy below these. Chosen to be lenient — the point is to
# catch collapse and unbound rewrites, not to grade style.
MIN_DISTINCT_RATIO = 0.85
MIN_REWRITE_JACCARD = 0.20
MIN_SENTIMENT_CORR = 0.75
MAX_TRUNCATION_RATE = 0.05
MIN_GENERATE_WORDS = 40  # below this the machine_generate output is a stub

# Word-set overlap (jaccard) is order-blind, so mechanical synonym substitution
# sails past it: swap "restaurant"->"venue", "food"->"cuisine" and the sets
# still match, the opinion still survives, every other metric still passes — but
# the text is not a rewrite, it is the source with words pasted over. The
# positional rate catches that: it is the share of the output's words sitting in
# an unchanged run against the source. Observed: genuine rewrites ~0.3-0.6,
# substitution spam ~0.9.
MAX_REWRITE_VERBATIM = 0.75

# `human_edit` is a copy-edit, so its overlap with the packet's source_text is
# a direct alignment probe: a faithful edit scores ~0.95+, and a run produced
# against a *different* packet scores near zero. This is the check that catches
# a packet regenerated after generation started — the failure is invisible in
# every other metric, because the output is a perfectly good copy-edit of a
# review that simply is not the one the packet now names.
MIN_EDIT_ALIGNMENT = 0.90

# Cross-record phrasing. "Distinct outputs" is not enough: a rule-based
# generator produced 99% distinct records while leaning on a shared phrase pool.
# The gate is *concentration* (share of 5-grams repeated 10+ times), which
# separates assembly from mere formulaic style: script 33.9%, genuine models
# <=1.5%, MiMo's own formulaic-but-genuine output 0.1%. The reuse rate is
# reported alongside it as a style statistic, not a gate.
MAX_GRAM_CONCENTRATION = 0.05
_CONCENTRATION_MIN = 10
_TEMPLATE_SAMPLE = 400
MIN_SENTIMENT_N = 8      # below this a correlation is not worth measuring
NOISY_SENTIMENT_N = 30   # below this, report it but flag it as indicative only


def _median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2] if xs else 0.0


def _template_reuse(texts: list[str], n: int = 5, sample: int = _TEMPLATE_SAMPLE) -> dict:
    """Cross-record phrasing statistics; `concentration` is the gate."""
    from collections import Counter

    grams: Counter = Counter()
    for text in texts[:sample]:
        words = [w.lower() for w in _words(text)]
        grams.update({" ".join(words[i : i + n]) for i in range(len(words) - n + 1)})
    if not grams:
        return {"n_records": 0, "reuse": 0.0, "concentration": 0.0, "top": []}
    total = len(grams)
    return {
        "n_records": min(len(texts), sample),
        "reuse": round(sum(1 for c in grams.values() if c > 1) / total, 4),
        "concentration": round(
            sum(1 for c in grams.values() if c >= _CONCENTRATION_MIN) / total, 4),
        "top": [[c, g] for g, c in grams.most_common(5) if c > 1],
    }


def _verbatim_rate(source: str, output: str) -> float:
    """Share of the output's words sitting in an unchanged run against source.

    Order-sensitive, unlike jaccard — which is the whole point: synonym
    substitution preserves the word set while rewriting almost nothing.
    """
    import difflib

    a, b = [w.lower() for w in _words(source)], [w.lower() for w in _words(output)]
    if not b:
        return 0.0
    same = sum(
        i2 - i1
        for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b).get_opcodes()
        if tag == "equal"
    )
    return same / len(b)


def _sentiment_pairs(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """(pearson r, mean absolute error) between source and output P(fake)."""
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from app.preprocess import preprocess

    tok = AutoTokenizer.from_pretrained(str(tc.OUTPUT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(tc.OUTPUT_DIR))
    model.eval()

    def score(texts: list[str]):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = [preprocess(t) for t in texts[i : i + 16]]
                enc = tok(batch, truncation=True, padding="max_length",
                          max_length=tc.MAX_LENGTH, return_tensors="pt")
                out.append(torch.softmax(model(**enc).logits.float(), -1)[:, 1].numpy())
        return np.concatenate(out)

    src, out = score([a for a, _ in pairs]), score([b for _, b in pairs])
    return float(np.corrcoef(src, out)[0, 1]), float(np.abs(src - out).mean())


def check(results_dir: Path, tasks_path: Path, do_sentiment: bool) -> dict:
    tasks = {}
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tasks[rec["custom_id"]] = rec

    rows, unparseable = [], 0
    for path in sorted(results_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    unparseable += 1
                    continue
                task = tasks.get(rec.get("custom_id"))
                if task is None:
                    continue
                text, _ = _extract_review_text(rec)
                if text:
                    rows.append((task, text))

    report: dict = {
        "n_records": len(rows),
        "unparseable_lines": unparseable,
        "distinct_ratio": round(len({t for _, t in rows}) / max(1, len(rows)), 4),
        "duplicate_examples": [
            [count, text[:60]]
            for text, count in Counter(t[:80] for _, t in rows).most_common(3)
            if count > 1
        ],
        "conditions": {},
    }

    for cond in ["machine_rewrite", "machine_generate", "human_edit"]:
        v = [(t, x) for t, x in rows if t["condition"] == cond]
        if not v:
            continue
        # Only count an unclean ending as truncation when the *source* ended
        # cleanly. Real Yelp reviews trail off ("JUST FYI*****", ":P"), and a
        # copy-edit that reproduces that is faithful, not truncated.
        truncated = [
            x for t, x in v
            if not _ENDS_CLEAN.search(x.strip())
            and not (t.get("source_text") and not _ENDS_CLEAN.search(t["source_text"].strip()))
        ]
        block: dict = {
            "n": len(v),
            "rejects": dict(Counter(_validate(x, t) for t, x in v)),
            "out_words_median": _median([len(_words(x)) for _, x in v]),
            "truncated": round(len(truncated) / len(v), 4),
        }
        with_src = [(t, x) for t, x in v if t.get("source_text")]
        if with_src:
            ratios = [len(_words(x)) / max(1, len(_words(t["source_text"])))
                      for t, x in with_src]
            jacs = [_jaccard(_words(x), _words(t["source_text"])) for t, x in with_src]
            block["len_ratio_median"] = round(_median(ratios), 3)
            block["jaccard_median"] = round(_median(jacs), 3)
            block["jaccard_below_0.15"] = round(
                sum(1 for j in jacs if j < 0.15) / len(jacs), 4
            )
            verb = sorted(_verbatim_rate(t["source_text"], x) for t, x in with_src)
            block["verbatim_median"] = round(_median(verb), 3)
            block["verbatim_p10"] = round(verb[len(verb) // 10], 3)
        report["conditions"][cond] = block

    if do_sentiment and tc.OUTPUT_DIR.exists():
        rw = [(t["source_text"], x) for t, x in rows
              if t["condition"] == "machine_rewrite" and t.get("source_text")]
        if len(rw) >= MIN_SENTIMENT_N:
            sample = rw[:400]
            corr, mae = _sentiment_pairs(sample)
            block = report["conditions"]["machine_rewrite"]
            block["sentiment_corr"] = round(corr, 3)
            block["sentiment_mae"] = round(mae, 3)
            # A pilot run has too few rewrites for r to be precise. Say so,
            # rather than letting a noisy number read as a verdict.
            block["sentiment_n"] = len(sample)
            block["sentiment_indicative_only"] = len(sample) < NOISY_SENTIMENT_N

    report["template_reuse"] = {
        cond: _template_reuse([x for t, x in rows if t["condition"] == cond])
        for cond in sorted({t["condition"] for t, _ in rows})
    }
    report["verdict"] = _verdict(report)
    return report


def _verdict(report: dict) -> dict:
    """Turn the metrics into pass/fail with the reason spelled out."""
    fails = []
    if report["distinct_ratio"] < MIN_DISTINCT_RATIO:
        fails.append(
            f"template collapse: only {report['distinct_ratio']:.0%} of records are "
            f"distinct (need >={MIN_DISTINCT_RATIO:.0%})"
        )
    rw = report["conditions"].get("machine_rewrite", {})
    if rw.get("jaccard_median", 1) < MIN_REWRITE_JACCARD:
        fails.append(
            f"rewrites are not bound to their source: median word overlap "
            f"{rw['jaccard_median']:.2f} (need >={MIN_REWRITE_JACCARD})"
        )
    if rw.get("verbatim_median", 0) > MAX_REWRITE_VERBATIM:
        fails.append(
            f"rewrites are synonym substitution, not rewriting: {rw['verbatim_median']:.0%} "
            f"of output words sit in unchanged runs (need <={MAX_REWRITE_VERBATIM:.0%}); "
            f"best decile {rw.get('verbatim_p10', 0):.0%}"
        )
    if "sentiment_corr" in rw and rw["sentiment_corr"] < MIN_SENTIMENT_CORR:
        # At pilot scale this is a signal, not proof — say which it is.
        scale = ("indicative only at this sample size"
                 if rw.get("sentiment_indicative_only") else "confirmed at scale")
        fails.append(
            f"rewrites do not preserve the source opinion: r="
            f"{rw['sentiment_corr']:.2f} (need >={MIN_SENTIMENT_CORR}) — {scale}"
        )
    he = report["conditions"].get("human_edit", {})
    if he.get("len_ratio_median", 1) < 0.9:
        fails.append(
            f"human_edit is not a copy-edit: output is "
            f"{he['len_ratio_median']:.0%} of source length (need >=90%)"
        )
    if he and he.get("jaccard_median", 1) < MIN_EDIT_ALIGNMENT:
        fails.append(
            f"run does not match this packet: human_edit overlap with the "
            f"packet's source_text is {he['jaccard_median']:.2f} "
            f"(need >={MIN_EDIT_ALIGNMENT}). Check the run was produced against "
            f"this packet and not an earlier regeneration of it."
        )
    gen = report["conditions"].get("machine_generate", {})
    if gen.get("out_words_median", 999) < MIN_GENERATE_WORDS:
        fails.append(
            f"machine_generate output is a stub: median {gen['out_words_median']} "
            f"words (need >={MIN_GENERATE_WORDS})"
        )
    for cond, b in report["conditions"].items():
        if b.get("truncated", 0) > MAX_TRUNCATION_RATE:
            fails.append(
                f"{cond}: {b['truncated']:.0%} of outputs end mid-sentence where "
                f"the source did not (need <={MAX_TRUNCATION_RATE:.0%})"
            )
    for cond, t in report.get("template_reuse", {}).items():
        if t["concentration"] > MAX_GRAM_CONCENTRATION:
            top = "; ".join(f"x{c} {g!r}" for c, g in t["top"][:2])
            fails.append(
                f"{cond}: {t['concentration']:.1%} of 5-grams repeat "
                f"{_CONCENTRATION_MIN}+ times (need <={MAX_GRAM_CONCENTRATION:.0%}) "
                f"— text looks assembled from a shared phrase pool rather than "
                f"written per record. Top: {top}"
            )
    if report["unparseable_lines"]:
        fails.append(f"{report['unparseable_lines']} unparseable lines")
    return {"usable": not fails, "problems": fails}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Health-check a generator run.")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--tasks", type=Path, default=PACKET_DIR / "tasks.jsonl")
    p.add_argument("--no-sentiment", action="store_true",
                   help="skip the (slower) opinion-preservation check")
    p.add_argument("--json", action="store_true", help="emit raw JSON")
    args = p.parse_args(argv)

    if not args.results.is_dir():
        raise SystemExit(f"Not a directory: {args.results}")

    report = check(args.results, args.tasks, not args.no_sentiment)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"== {args.generator}: {report['n_records']} records ==\n")
    print(f"  distinct outputs     {report['distinct_ratio']:.1%}")
    for cond, t in report.get("template_reuse", {}).items():
        print(f"  phrasing             {cond:18s} reuse {t['reuse']:5.1%}  "
              f"concentration {t['concentration']:5.2%}"
              f"   (gate: concentration <= {MAX_GRAM_CONCENTRATION:.0%})")
    for count, text in report["duplicate_examples"]:
        print(f"      x{count}  {text!r}")
    for cond, b in report["conditions"].items():
        print(f"\n  -- {cond} (n={b['n']}) --")
        if "len_ratio_median" in b:
            print(f"     length vs source  {b['len_ratio_median']:.2f}x"
                  f"   overlap {b['jaccard_median']:.2f}"
                  f"   ({b['jaccard_below_0.15']:.0%} below 0.15)")
            print(f"     verbatim runs     {b['verbatim_median']:.2f} median"
                  f"   {b['verbatim_p10']:.2f} at best decile"
                  f"   (rewrites must be <={MAX_REWRITE_VERBATIM:.2f})")
        print(f"     out words (med)   {b['out_words_median']}")
        print(f"     truncated         {b['truncated']:.1%}"
              f"   (ends mid-sentence where source did not)")
        if "sentiment_corr" in b:
            note = (f"  (n={b['sentiment_n']}, indicative only)"
                    if b.get("sentiment_indicative_only") else "")
            print(f"     opinion preserved r={b['sentiment_corr']:.3f}"
                  f"  MAE={b['sentiment_mae']:.3f}{note}")
        nz = {k: v for k, v in b["rejects"].items() if k != "None"}
        print(f"     rejects           {nz or 'none'}")

    v = report["verdict"]
    print(f"\n  VERDICT: {'USABLE' if v['usable'] else 'NOT USABLE'}")
    for problem in v["problems"]:
        print(f"    - {problem}")


if __name__ == "__main__":
    main()
