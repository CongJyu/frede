"""Validate and normalise Stage 1 generator output into a training file.

Accepts the documented `{"custom_id", "text"}` contract, but also unwraps
Anthropic and OpenAI *batch result* files directly, so a raw batch dump can be
fed in without post-processing.

Nothing is dropped silently: every record either lands in the output or is
counted under a named reject reason in the printed report and in the sidecar
`*.report.json`.

Usage:
    python -m scripts.ingest_gen_results \
        --results results/claude-opus-5.jsonl --generator claude-opus-5
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from app.preprocess import normalize_escapes
from training import config as tc

PACKET_DIR = tc.DATA_DIR / "gen_packet"
OUT_DIR = tc.DATA_DIR / "gen_pairs"

# Length as a multiple of the source, per condition. Loose bounds: the point is
# to catch a generator that ignored the task, not to police style.
LEN_BOUNDS = {
    "machine_rewrite": (0.35, 2.5),
    "human_edit": (0.80, 1.20),
    "machine_generate": (0.0, 0.0),  # absolute bounds used instead
}
GENERATE_WORDS = (35, 260)

# `human_edit` must be a copy-edit, so word overlap with the source stays high.
# A low-overlap "edit" is really a rewrite wearing the control's label, which
# would quietly corrupt the hard-negative set.
EDIT_MIN_JACCARD = 0.70
REWRITE_MAX_JACCARD = 0.92
# Order-sensitive companion to jaccard. Synonym substitution keeps the word set
# (so jaccard passes) while rewriting almost nothing; this is the share of the
# output's words sitting in an unchanged run against the source, which catches
# it. Genuine rewrites sit ~0.3-0.6, substitution ~0.9.
REWRITE_MAX_VERBATIM = 0.75

# Template detection, measured as *concentration* rather than reuse rate.
#
# A plain "share of n-grams seen more than once" conflates two different things:
# text assembled from a small phrase pool, and text that is individually written
# but formulaic. Both raise it, so it cannot gate on its own. What separates
# them is how heavily the repeats concentrate — a template leans on few phrases
# many times, a formulaic model reuses many phrases a few times:
#
#   condition            grams seen >=10x      reuse
#   script (from scratch)      33.94%          51.7%
#   MiMo (genuine)              0.11%          17.5%
#   DeepSeek (genuine)          0.01%           6.7%
#
# The source-anchored conditions are ~0 for the script too (0.01% rewrite,
# 0.00% edit) because each follows its own source, which is why this only
# catches the from-scratch condition — correct, since that is where a template
# can operate. The reuse rate is still reported, as a style statistic relevant
# to single-generator risk, but it does not gate.
MAX_GRAM_CONCENTRATION = 0.05
CONCENTRATION_MIN_COUNT = 10
TEMPLATE_SAMPLE = 400

# `human_edit` overlap with the packet's `source_text`, used as a run-alignment
# probe: a faithful copy-edit scores ~0.95, a run made against a different
# packet scores near zero.
MIN_EDIT_ALIGNMENT = 0.90

# Deliberately narrow: only unambiguous model self-reference. Broader refusal
# phrasing is not usable here — measured against 2400 real Yelp reviews,
# "I cannot help", "I can't help but" and "I must refuse" are all ordinary
# customer English and flagged real people. Refusals that dodge these patterns
# are caught by the length checks instead.
# An output ending is only judged *against its source*: real Yelp reviews trail
# off ("JUST FYI*****", ":P", "Thanks guys!!!"), and a faithful copy-edit that
# reproduces that is correct, not truncated. A mismatch where the source ended
# cleanly and the output did not is the reliable truncation signature.
_ENDS_CLEAN = re.compile(r"""[.!?"')\]`*]\s*$""")

_META_RE = re.compile(
    r"\bas an ai\b|\bas a language model\b|\bi am an ai\b|"
    r"\bi(?:'m| am) (?:just )?(?:a|an) (?:ai|language model)\b|"
    r"\blanguage model\b|"
    r"\bi don't have (?:personal )?(?:experiences|opinions)\b",
    re.I,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0



def _verbatim_rate(source: str, output: str) -> float:
    """Share of `output`'s words sitting in an unchanged run against `source`.

    Order-sensitive, unlike jaccard — which is the point: synonym substitution
    preserves the word set while rewriting almost nothing.
    """
    import difflib

    a, b = _words(source), _words(output)
    if not b:
        return 0.0
    same = sum(
        i2 - i1
        for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b).get_opcodes()
        if tag == "equal"
    )
    return same / len(b)


def _template_stats(texts: list[str], n: int = 5, sample: int = TEMPLATE_SAMPLE) -> dict:
    """Cross-record phrasing statistics.

    `concentration` is the share of n-grams repeated at least
    `CONCENTRATION_MIN_COUNT` times — the gate. `reuse` is the share repeated
    more than once — reported, not gated; it measures formulaic style, which is
    a property of the generator rather than proof of assembly.
    """
    grams: Counter = Counter()
    for text in texts[:sample]:
        words = _words(text)
        grams.update({" ".join(words[i : i + n]) for i in range(len(words) - n + 1)})
    if not grams:
        return {"n_records": 0, "reuse": 0.0, "concentration": 0.0, "top": []}
    total = len(grams)
    return {
        "n_records": min(len(texts), sample),
        "reuse": round(sum(1 for c in grams.values() if c > 1) / total, 4),
        "concentration": round(
            sum(1 for c in grams.values() if c >= CONCENTRATION_MIN_COUNT) / total, 4),
        "top": [[c, g] for g, c in grams.most_common(5) if c > 1],
    }


def run_level_failures(out_records: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Checks that only exist across records, so they run after the loop.

    Returns (run_wide, per_condition). The split matters: a misaligned packet
    invalidates *everything* the run produced, while a condition that fails on
    its own — substitution rewrites, say — should cost exactly that condition
    and not the good data beside it. Blocking wholesale would throw away usable
    records; blocking nothing would admit the bad ones.
    """
    run_wide: list[str] = []
    per_condition: dict[str, str] = {}
    by_cond: dict[str, list[dict]] = {}
    for r in out_records:
        by_cond.setdefault(r["condition"], []).append(r)

    for cond, recs in sorted(by_cond.items()):
        stats = _template_stats([r["text"] for r in recs])
        if stats["concentration"] > MAX_GRAM_CONCENTRATION:
            top = "; ".join(f"x{c} {g!r}" for c, g in stats["top"][:2])
            per_condition[cond] = (
                f"{stats['concentration']:.1%} of 5-grams repeat "
                f"{CONCENTRATION_MIN_COUNT}+ times across records "
                f"(limit {MAX_GRAM_CONCENTRATION:.0%}) — looks assembled from a "
                f"shared phrase pool rather than written per record. Top: {top}"
            )

    edits = by_cond.get("human_edit", [])
    if edits:
        jacs = sorted(_jaccard(_words(r["text"]), _words(r["source_text"]))
                      for r in edits if r["source_text"])
        if jacs and jacs[len(jacs) // 2] < MIN_EDIT_ALIGNMENT:
            run_wide.append(
                f"human_edit: median overlap with the packet's source_text is "
                f"{jacs[len(jacs) // 2]:.2f} (need >={MIN_EDIT_ALIGNMENT}) — this "
                f"run was probably produced against a different packet, so "
                f"nothing it contains can be trusted."
            )
    return run_wide, per_condition


def _extract_review_text(record: dict) -> tuple[str | None, str | None]:
    """Pull the review out of any supported result shape.

    Returns (text, note). `note` is non-None when the shape needed unwrapping
    in a way worth surfacing.
    """
    # Anthropic batch result
    if "result" in record and isinstance(record["result"], dict):
        res = record["result"]
        if res.get("type") != "succeeded":
            return None, f"anthropic:{res.get('type', 'unknown')}"
        blocks = (res.get("message") or {}).get("content") or []
        raw = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
    # OpenAI batch result
    elif "response" in record and isinstance(record["response"], dict):
        body = record["response"].get("body") or {}
        choices = body.get("choices") or []
        raw = (choices[0].get("message") or {}).get("content", "") if choices else ""
    elif isinstance(record.get("text"), str):
        raw = record["text"]
    elif isinstance(record.get("completion"), str):
        raw = record["completion"]
    else:
        return None, "no-text-field"

    raw = raw.strip()
    if not raw:
        return None, "empty"

    # The contract asks for a JSON object; unwrap it when present, but keep the
    # raw response when the model answered with the review directly.
    candidate = _FENCE_RE.sub("", raw).strip()
    if candidate.startswith("{"):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                return obj["text"].strip(), "unwrapped-json"
        except json.JSONDecodeError:
            # e.g. an unescaped newline inside the string — salvage the field.
            m = re.search(r'"text"\s*:\s*"(.*)"\s*\}?\s*$', candidate, re.S)
            if m:
                return m.group(1).encode().decode("unicode_escape", "ignore").strip(), \
                    "salvaged-json"
            return None, "bad-json"
    return raw, "raw-text"


def _resolve_results(paths: list[Path]) -> list[Path]:
    """Expand directories to their `*.jsonl`, so a shard output dir just works."""
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(p for p in path.glob("*.jsonl") if not p.name.startswith("_"))
            if not found:
                raise SystemExit(f"No .jsonl files in {path}")
            out.extend(found)
        elif path.exists():
            out.append(path)
        else:
            raise SystemExit(f"No such results path: {path}")
    return out


def ingest(results_paths: list[Path], generator: str, tasks_path: Path,
           force: bool = False) -> dict:
    tasks = {}
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tasks[rec["custom_id"]] = rec

    out_records, rejects, notes = [], Counter(), Counter()
    # Everything successfully extracted, including records the per-record
    # validator later rejects. Run-level checks must see these: a misaligned run
    # is caught by its rejected records disappearing, which would otherwise make
    # the surviving subset look perfectly aligned.
    extracted: list[dict] = []
    seen: set[str] = set()

    for path in results_paths:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    rejects["unparseable-line"] += 1
                    print(f"  ! {path.name}:{lineno} unparseable JSON line")
                    continue

                custom_id = rec.get("custom_id")
                if custom_id not in tasks:
                    rejects["unknown-custom_id"] += 1
                    continue
                if custom_id in seen:
                    rejects["duplicate-custom_id"] += 1
                    continue
                seen.add(custom_id)

                task = tasks[custom_id]
                text, note = _extract_review_text(rec)
                notes[note] += 1
                if text is None:
                    rejects[f"extract:{note}"] += 1
                    continue

                # Both sides get the same escape normalisation, so the Yelp
                # corpus's literal "\n" cannot masquerade as a class signal.
                text = normalize_escapes(text)
                source = normalize_escapes(task["source_text"]) \
                    if task["source_text"] else None
                extracted.append({"condition": task["condition"], "text": text,
                                  "source_text": source})
                reason = _validate(text, {**task, "source_text": source})
                if reason:
                    rejects[reason] += 1
                    continue

                out_records.append({
                    "pair_id": task["pair_id"],
                    "custom_id": custom_id,
                    "condition": task["condition"],
                    "split": task["split"],
                    "stars": task["stars"],
                    # machine_rewrite / machine_generate are machine-authored;
                    # human_edit is a copy-edited human review and stays label 0
                    # — it is the hard negative that stops the detector from
                    # simply learning "text with no typos = machine".
                    "label": 0 if task["condition"] == "human_edit" else 1,
                    "generator": generator,
                    "text": text,
                    "source_text": source,
                })

    # Run-level gate. A generation run can pass every per-record check and still
    # be unusable — a fully template-generated run did exactly that. Refusing to
    # write is the point: a NOT USABLE verdict should block, not become a caveat
    # attached to a file that then gets trained on anyway.
    run_wide, per_condition = run_level_failures(extracted)
    dropped_conditions: dict[str, str] = {}
    out_path = OUT_DIR / f"{generator}.jsonl"

    if run_wide and not force:
        detail = "\n  - ".join(run_wide)
        raise SystemExit(
            f"\nREFUSING to write {out_path} — the run failed "
            f"{len(run_wide)} run-level check(s):\n  - {detail}\n\n"
            f"Pass --force to write anyway (and expect downstream results to be "
            f"meaningless)."
        )

    if per_condition and not force:
        kept = [r for r in out_records if r["condition"] not in per_condition]
        for cond, why in sorted(per_condition.items()):
            n = sum(1 for r in out_records if r["condition"] == cond)
            dropped_conditions[cond] = why
            print(f"  ! dropping condition '{cond}' ({n} records): {why}")
        if not kept:
            raise SystemExit(
                f"\nREFUSING to write {out_path} — every condition failed its "
                f"run-level check. Pass --force to write anyway."
            )
        out_records = kept

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{generator}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_condition = Counter(r["condition"] for r in out_records)
    report = {
        "generator": generator,
        "results_files": [str(p) for p in results_paths],
        "n_tasks": len(tasks),
        "n_ingested": len(out_records),
        "coverage": round(len(out_records) / max(1, len(tasks)), 4),
        "by_condition": dict(by_condition),
        "by_stars": dict(sorted(Counter(r["stars"] for r in out_records).items())),
        "by_split": dict(Counter(r["split"] for r in out_records)),
        "extraction_notes": dict(notes),
        "rejects": dict(rejects),
        "dropped_conditions": dropped_conditions,
    }
    with open(out_path.with_suffix(".report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report, out_path


def _validate(text: str, task: dict) -> str | None:
    """Return a reject reason, or None if the record is usable."""
    words = _words(text)
    if len(words) < 10:
        return "too-short"
    if _META_RE.search(text):
        return "meta-commentary"

    condition = task["condition"]
    source = task["source_text"]

    if condition == "machine_generate":
        lo, hi = GENERATE_WORDS
        if not (lo <= len(words) <= hi):
            return "generate-length"
        return None

    src_words = _words(source or "")
    if not src_words:
        return "missing-source"

    ratio = len(words) / len(src_words)
    lo, hi = LEN_BOUNDS[condition]
    if not (lo <= ratio <= hi):
        return f"{condition}-length-ratio"

    # Truncation: the source finished its sentence and the output did not.
    if _ENDS_CLEAN.search(source.strip()) and not _ENDS_CLEAN.search(text.strip()):
        return "truncated"

    jac = _jaccard(words, src_words)
    if condition == "human_edit" and jac < EDIT_MIN_JACCARD:
        # Rewrote instead of copy-editing: would silently poison the control.
        return "edit-content-drift"
    if condition == "machine_rewrite" and jac > REWRITE_MAX_JACCARD:
        return "rewrite-too-close-to-source"
    if condition == "machine_rewrite" and _verbatim_rate(source, text) > REWRITE_MAX_VERBATIM:
        return "rewrite-is-synonym-substitution"
    return None



def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Ingest Stage 1 generator results.")
    p.add_argument("--results", type=Path, nargs="+", required=True,
                   help="result files, or a directory of per-shard *.jsonl outputs")
    p.add_argument("--generator", required=True,
                   help="stable tag for the generating model, e.g. deepseek-v4f-cline")
    p.add_argument("--tasks", type=Path, default=PACKET_DIR / "tasks.jsonl")
    p.add_argument("--force", action="store_true",
                   help="write even when run-level checks fail (downstream "
                        "results will be meaningless)")
    args = p.parse_args(argv)

    if not args.tasks.exists():
        raise SystemExit(f"No packet at {args.tasks}. Run scripts.make_gen_packet first.")

    report, out_path = ingest(
        _resolve_results(args.results), args.generator, args.tasks, args.force
    )
    print(f"\nIngested {report['n_ingested']}/{report['n_tasks']} "
          f"({report['coverage']:.1%} coverage) from {args.generator}")
    print(f"  by condition : {report['by_condition']}")
    print(f"  by stars     : {report['by_stars']}")
    print(f"  by split     : {report['by_split']}")
    print(f"  extraction   : {report['extraction_notes']}")
    if report["rejects"]:
        print(f"  REJECTED     : {report['rejects']}")
    else:
        print("  rejected     : none")
    print(f"\nPairs -> {out_path}\nReport -> {out_path.with_suffix('.report.json')}")


if __name__ == "__main__":
    main()
