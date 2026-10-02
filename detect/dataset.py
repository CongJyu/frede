"""Stage 2 dataset assembly.

Three design rules, each of which exists to keep the "add a second generator
later" step a parameter change rather than a rewrite:

1. **The human class is generator-independent.** It is drawn from the packet's
   own `source_text` (real Yelp reviews), never from a generator's `human_edit`
   output. Swapping or adding a generator therefore never perturbs the negative
   class, so two runs stay comparable.

2. **Splits are keyed to `custom_id`, not to position or generator.** The frozen
   packet already assigns `train`/`dev` per custom_id, so a second generator
   running the same packet inherits the identical split automatically. Train on
   generator A's train records and test on generator B's dev records — that is
   the cross-generator measurement, and it needs no extra machinery.

3. **`human_edit` is a probe, not training data.** It is human-authored text
   that a model has copy-edited. Keeping it out of training does two things: the
   human class stays plainly human, and the probe measures the failure mode that
   matters — how often the detector flags a real person's review just because it
   is now clean and well-punctuated. Training on it would hide that.

`human_eval.jsonl` is the second probe: raw Yelp reviews never used as a pair
source, for the headline false-positive rate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from training import config as tc

PAIRS_DIR = tc.DATA_DIR / "gen_pairs"
PACKET_DIR = tc.DATA_DIR / "gen_packet"

MACHINE_CONDITIONS = ("machine_generate", "machine_rewrite")


@dataclass(frozen=True)
class Record:
    record_id: str
    text: str
    label: int            # 1 = machine, 0 = human
    split: str            # "train" | "dev"
    condition: str
    generator: str | None  # None for human records
    custom_id: str | None
    stars: int | None


def _load_packet(packet_dir: Path) -> dict[str, dict]:
    with open(packet_dir / "tasks.jsonl", encoding="utf-8") as f:
        return {r["custom_id"]: r for r in (json.loads(line) for line in f)}


def build(
    generators: list[str],
    packet_dir: Path = PACKET_DIR,
    pairs_dir: Path = PAIRS_DIR,
) -> tuple[list[Record], list[Record], dict[str, list[Record]]]:
    """Return (train, test, probes).

    `generators` names the Stage 1 runs to draw the machine class from. Passing
    two tags yields the cross-generator comparison with no other change; their
    records share custom_ids, so both land in the same split.
    """
    packet = _load_packet(packet_dir)

    # ---- human class: real Yelp reviews, straight from the packet ----------
    # Deduplicated by text: the rewrite and edit conditions draw from disjoint
    # source pools, but dedupe anyway so a future packet change can't silently
    # double-count a review.
    human: list[Record] = []
    seen_text: set[str] = set()
    for custom_id, rec in packet.items():
        text = rec.get("source_text")
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        human.append(Record(
            record_id=f"human::{custom_id}",
            text=text,
            label=0,
            split=rec["split"],
            condition="yelp_source",
            generator=None,
            custom_id=custom_id,
            stars=rec.get("stars"),
        ))

    # ---- machine class: generator output ----------------------------------
    machine: list[Record] = []
    for tag in generators:
        path = pairs_dir / f"{tag}.jsonl"
        if not path.exists():
            raise SystemExit(f"No pairs file for generator {tag!r} at {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                if p["condition"] not in MACHINE_CONDITIONS:
                    continue  # human_edit is a probe, never a machine positive
                task = packet.get(p["custom_id"])
                if task is None:
                    continue
                machine.append(Record(
                    record_id=f"{tag}::{p['custom_id']}",
                    text=p["text"],
                    label=1,
                    split=p["split"],
                    condition=p["condition"],
                    generator=tag,
                    custom_id=p["custom_id"],
                    stars=p.get("stars"),
                ))

    # ---- probes: never trained on -----------------------------------------
    probes: dict[str, list[Record]] = {}

    edit_path = pairs_dir / f"{generators[0]}.jsonl"
    edits = []
    if edit_path.exists():
        with open(edit_path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                if p["condition"] != "human_edit":
                    continue
                # A generator that returned the review untouched tells us nothing
                # about whether polish triggers a false positive — and its text
                # is identical to the source, which is *in* the training set.
                # Only records the model actually changed are informative.
                changed = p["text"].strip() != (p.get("source_text") or "").replace(
                    "\\n", "\n").strip()
                edits.append(Record(
                    record_id=f"probe::human_edit::{p['custom_id']}",
                    text=p["text"], label=0, split="probe",
                    condition="human_edit_edited" if changed else "human_edit_unchanged",
                    generator=p["generator"],
                    custom_id=p["custom_id"], stars=p.get("stars"),
                ))
    probes["human_edit_polished"] = edits

    eval_path = packet_dir / "human_eval.jsonl"
    raw = []
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                h = json.loads(line)
                raw.append(Record(
                    record_id=f"probe::human_eval::{h['pair_id']}",
                    text=h["text"], label=0, split="probe",
                    condition="yelp_human_heldout", generator=None,
                    custom_id=h["pair_id"], stars=h.get("stars"),
                ))
    probes["human_heldout_raw"] = raw

    train = [r for r in human + machine if r.split == "train"]

    # A generator that emits the same review twice puts an exact copy of a
    # training text into the test set, which scores as a free hit and inflates
    # the result. Observed in practice at ~0.7% (residual template collapse).
    train_texts = {r.text for r in train}
    test, dropped = [], 0
    for r in (r for r in human + machine if r.split == "dev"):
        if r.text in train_texts:
            dropped += 1
            continue
        test.append(r)
    if dropped:
        print(f"[dataset] dropped {dropped} test records duplicating a train text")

    return train, test, probes


def summarize(records: list[Record]) -> dict:
    from collections import Counter

    return {
        "n": len(records),
        "label": dict(Counter(r.label for r in records)),
        "condition": dict(Counter(r.condition for r in records)),
        "generator": dict(Counter(r.generator or "human" for r in records)),
    }
