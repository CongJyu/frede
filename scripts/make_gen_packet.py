"""Build the Stage 1 generation packet for an external LLM.

Samples real Yelp reviews, renders a fully-formed prompt per review, and writes
the packet an external generator consumes as small per-shard work units.

Design notes live in `prompts/generate_machine_reviews.md`; the operator brief
handed to a generating agent is `prompts/GENERATION_BRIEF.md`. The two design
points that matter most: every record pins its star rating, and `human_edit`
exists as a negative control so Stage 2 can tell "machine-written" apart from
"has no typos".

Sampling is fully determined by `--seed` and is independent of `--shard-size`
and `--out-dir`, so re-sharding (e.g. a smaller-shard retry after a generator
under-performed) reproduces byte-identical records with the same `custom_id`s.

Usage:
    python -m scripts.make_gen_packet --n-per-condition 1000 --n-human-eval 400
    python -m scripts.make_gen_packet --n-per-condition 1000 --shard-size 10 \\
        --out-dir data/gen_packet_small
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Below this, a review has no style to speak of ("Great food!"). Rewriting or
# copy-editing it produces a record that cannot be judged or learned from.
MIN_SOURCE_WORDS = 25
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

from detect.stage0 import load_yelp_raw
from training import config as tc

OUT_DIR = tc.DATA_DIR / "gen_packet"
SYSTEM_PROMPT = (
    "You are generating synthetic restaurant and local-business reviews for an "
    "academic dataset on machine-generated text detection. Follow the user's "
    "formatting instructions exactly. Never mention that you are an AI, never "
    "mention this task, and never add commentary, headings, or explanation "
    "outside the requested JSON."
)

# Deliberately broad: yelp_review_full is all local businesses, not only
# restaurants (the corpus contains doctors, mechanics, etc.).
BUSINESS_TYPES = [
    "a casual restaurant", "a fine-dining restaurant", "a coffee shop",
    "a pizza place", "a sushi restaurant", "a burger joint", "a bakery",
    "a food truck", "a bar", "an ice cream shop", "a dentist's office",
    "an auto repair shop", "a hair salon", "a gym", "a hardware store",
    "a hotel", "a plumber", "a veterinarian", "a bookstore", "a dry cleaner",
]

_TAIL = (
    '\n\nRespond with exactly one JSON object and nothing else, no code fences:\n'
    '{"custom_id": "%s", "text": "<the review>"}'
)

REWRITE_TEMPLATE = """Rewrite the review below in your own words.

Requirements:
- Keep the SAME star rating and the SAME overall opinion. A negative review stays negative; a glowing review stays glowing. Do not soften or intensify it.
- Keep every concrete detail: what was ordered or done, the staff, the price, the wait, specific incidents. Do not invent new facts and do not drop the specifics. If the source names no business, do not add one.
- Write it the way you naturally write as an assistant. Do not deliberately imitate a human, and do not exaggerate or caricature "AI style".
- Do not copy more than a few consecutive words from the source. This must be a genuine rewrite, not a light edit.
- Aim for a similar length to the source.

Source review ({stars} out of 5 stars):
---
{source}
---"""

GENERATE_TEMPLATE = """Write a review of {business}, rated {stars} out of 5 stars.

Requirements:
- The review must clearly justify a {stars}-star rating. A 1-star review should read as genuinely bad; a 5-star review as genuinely great.
- Invent a plausible specific business and concrete details — dishes, service, prices, atmosphere, how long you waited — so it reads like a real customer wrote it.
- Write it the way you naturally write as an assistant. Do not deliberately imitate a human, and do not exaggerate or caricature "AI style".
- Between 60 and 160 words.
- Do not use star emoji or a numeric rating in the text."""

EDIT_TEMPLATE = """Below is a real customer review. Make ONLY mechanical corrections.

Requirements:
- Fix spelling, grammar, capitalisation and punctuation.
- Do NOT change the wording, vocabulary, tone, sentence structure, sentence order, or length.
- Do NOT make it sound more polished, more professional, or more fluent.
- Keep every factual detail and the same overall opinion.
- The result must read exactly like the same person's writing, just with the errors fixed. If the source has no errors, return it unchanged.

Source review ({stars} out of 5 stars):
---
{source}
---"""


def render_prompt(condition: str, stars: int, custom_id: str,
                  source: str | None, business: str | None) -> str:
    if condition == "machine_rewrite":
        body = REWRITE_TEMPLATE.format(stars=stars, source=source)
    elif condition == "machine_generate":
        body = GENERATE_TEMPLATE.format(stars=stars, business=business)
    elif condition == "human_edit":
        body = EDIT_TEMPLATE.format(stars=stars, source=source)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return body + (_TAIL % custom_id)


def _stratified_sample(df, n_per_star: int, rng: np.random.RandomState):
    """Equal counts per star level, so sentiment is a controlled variable."""
    picks = []
    for star in sorted(df["stars"].unique()):
        pool = df.index[df["stars"] == star].to_numpy()
        take = min(n_per_star, len(pool))
        picks.extend(rng.choice(pool, size=take, replace=False).tolist())
    rng.shuffle(picks)
    return df.loc[picks].reset_index(drop=True)


def build(args) -> dict:
    rng = np.random.RandomState(args.seed)

    # Each condition gets `n_per_condition` records, spread evenly over the five
    # star levels so rating is a controlled variable rather than a confound.
    per_star = max(1, args.n_per_condition // 5)
    need = per_star * 5

    # `machine_rewrite` and `human_edit` draw from *disjoint* source sets: each
    # pair then rests on a different real review, which widens the human-text
    # distribution the detector sees instead of reusing the same 1000.
    # The human-eval pool is disjoint from both, so no review can be both a pair
    # source and part of the false-positive measurement.
    n_sources = need * 2 + 200  # +200 absorbs the dedupe + length drops
    pool = load_yelp_raw(n_sources * 2 + args.n_human_eval, args.seed)
    # Very short reviews ("Great food!") carry no style signal, so a rewrite or
    # copy-edit of one is unusable — drop them here rather than spending a
    # generation call and rejecting the result later.
    pool = pool[pool["text"].map(lambda t: len(_WORD_RE.findall(t)) >= MIN_SOURCE_WORDS)]
    pool = pool.reset_index(drop=True)
    if len(pool) < n_sources + args.n_human_eval:
        raise SystemExit("Not enough long-enough Yelp reviews; lower --n-per-condition.")
    human_eval = pool.iloc[n_sources : n_sources + args.n_human_eval].reset_index(drop=True)
    sources = _stratified_sample(pool.iloc[:n_sources], per_star * 2, rng)
    if len(sources) < need * 2:
        raise SystemExit(
            f"Only {len(sources)} source reviews available, need {need * 2} "
            f"({per_star * 2} per star level). Lower --n-per-condition."
        )
    edit_sources = sources.iloc[need : need * 2].reset_index(drop=True)

    records, pid = [], 0
    for condition, src in [("machine_rewrite", sources), ("human_edit", edit_sources)]:
        for _, row in src.iloc[:need].iterrows():
            custom_id = f"p{pid:06d}_{condition}"
            records.append({
                "custom_id": custom_id,
                "pair_id": f"p{pid:06d}",
                "condition": condition,
                "split": "dev" if rng.rand() < args.dev_frac else "train",
                "stars": int(row["stars"]),
                "source_text": row["text"],
                "business": None,
                "prompt": render_prompt(condition, int(row["stars"]), custom_id,
                                        row["text"], None),
            })
            pid += 1

    # `machine_generate` needs a rating and a business type, never the source —
    # handing it the source text would just make it another rewrite. Ratings are
    # drawn star-balanced so the condition stays comparable to the other two.
    gen_stars = [s for s in range(1, 6) for _ in range(per_star)]
    rng.shuffle(gen_stars)
    for stars in gen_stars:
        custom_id = f"p{pid:06d}_machine_generate"
        business = BUSINESS_TYPES[rng.randint(len(BUSINESS_TYPES))]
        records.append({
            "custom_id": custom_id,
            "pair_id": f"p{pid:06d}",
            "condition": "machine_generate",
            "split": "dev" if rng.rand() < args.dev_frac else "train",
            "stars": stars,
            "source_text": None,
            "business": business,
            "prompt": render_prompt("machine_generate", stars, custom_id, None,
                                    business),
        })
        pid += 1

    rng.shuffle(records)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    # A packet is a frozen contract: generator output is keyed by custom_id, so
    # regenerating one after a run has started silently re-points those ids at
    # different reviews and invalidates the whole run. Refuse unless forced.
    if (out_dir / "tasks.jsonl").exists() and not args.force:
        raise SystemExit(
            f"{out_dir} already exists. Regenerating it would re-map custom_ids "
            f"and invalidate any generator output already produced against it.\n"
            f"Use a different --out-dir, or --force if no run depends on it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out_dir / "system_prompt.txt", "w", encoding="utf-8") as f:
        f.write(SYSTEM_PROMPT + "\n")
    _write_shards(records, args.shard_size, out_dir)

    # Human held-out set: never a pair source, used only to measure how often
    # the Stage 2 detector flags real people.
    with open(out_dir / "human_eval.jsonl", "w", encoding="utf-8") as f:
        for i, row in human_eval.iterrows():
            f.write(json.dumps({
                "pair_id": f"h{i:06d}", "stars": int(row["stars"]),
                "source": "yelp_human", "text": row["text"],
            }, ensure_ascii=False) + "\n")

    if args.vendor_files:
        _write_requests(records, args, out_dir)
    return _write_manifest(records, human_eval, args, out_dir)


def _write_shards(records: list[dict], shard_size: int, out_dir: Path) -> None:
    """Split the packet into small per-turn work units.

    Handing a coding agent one 3000-record file invites it to truncate, skip, or
    lose its place. Small shards make each unit of work independently
    resumable: shard N's output file existing means shard N is done.
    """
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("shard_*.jsonl"):
        old.unlink()
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    index = []
    for i in range(0, len(lines), shard_size):
        chunk = lines[i : i + shard_size]
        name = f"shard_{i // shard_size:03d}"
        (shard_dir / f"{name}.jsonl").write_text("\n".join(chunk) + "\n", encoding="utf-8")
        index.append({"shard": name, "n": len(chunk),
                      "first": json.loads(chunk[0])["custom_id"],
                      "last": json.loads(chunk[-1])["custom_id"]})
    (shard_dir / "_index.json").write_text(
        json.dumps({"shard_size": shard_size, "n_shards": len(index),
                    "shards": index}, indent=2),
        encoding="utf-8",
    )
    print(f"  shards       : {len(index)} files of <={shard_size} -> {shard_dir}")


def _write_requests(records: list[dict], args, out_dir: Path) -> None:
    """Pre-wrapped batch request files, so no re-templating is needed."""
    with open(out_dir / "requests.anthropic.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({
                "custom_id": rec["custom_id"],
                # NB: no temperature/top_p — removed on Opus 5 / Sonnet 5 /
                # Fable 5 / Opus 4.8+ and rejected with a 400.
                "params": {
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": rec["prompt"]}],
                },
            }, ensure_ascii=False) + "\n")

    with open(out_dir / "requests.openai.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({
                "custom_id": rec["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": rec["prompt"]},
                    ],
                },
            }, ensure_ascii=False) + "\n")


def _write_manifest(records, human_eval, args, out_dir: Path) -> dict:
    def by(key: str) -> dict:
        return dict(sorted(Counter(r[key] for r in records).items(), key=str))

    manifest = {
        "packet_version": 1,
        "description": "Stage 1 paired machine-review generation packet",
        "seed": args.seed,
        "dev_frac": args.dev_frac,
        "model_hint": args.model,
        "max_tokens": args.max_tokens,
        "n_records": len(records),
        "n_human_eval": len(human_eval),
        "shard_size": args.shard_size,
        "n_shards": -(-len(records) // args.shard_size),  # ceil
        "by_condition": by("condition"),
        "by_split": by("split"),
        "by_stars": by("stars"),
        "notes": [
            "prompt is pre-rendered; send verbatim, do not re-template",
            "run the whole packet through >=2 different models for the "
            "cross-generator split",
            "shards/ holds the work units; one output file per shard makes the "
            "run resumable",
        ],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the Stage 1 generation packet.")
    p.add_argument("--n-per-condition", type=int, default=500,
                   help="records per condition, split evenly across the 5 star levels")
    p.add_argument("--n-human-eval", type=int, default=400,
                   help="held-out human reviews for false-positive measurement")
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--model", default="claude-opus-5",
                   help="model id written into the vendor batch request files")
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--shard-size", type=int, default=50,
                   help="records per shard file (one unit of agent work)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing packet (invalidates generator "
                        "output already keyed to it)")
    p.add_argument("--out-dir", default=None,
                   help="write the packet here instead of data/gen_packet "
                        "(used to build a smaller-shard retry without disturbing "
                        "an in-flight run's shard numbering)")
    p.add_argument("--vendor-files", action="store_true",
                   help="also emit requests.anthropic.jsonl / requests.openai.jsonl "
                        "(only needed for the Batch APIs)")
    args = p.parse_args(argv)

    manifest = build(args)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    print(f"Packet written to {out_dir}")
    print(f"  records      : {manifest['n_records']}  {manifest['by_condition']}")
    print(f"  split        : {manifest['by_split']}")
    print(f"  stars        : {manifest['by_stars']}")
    print(f"  human eval   : {manifest['n_human_eval']}")
    print(f"  system prompt: {out_dir / 'system_prompt.txt'}")
    print(f"  tasks        : {out_dir / 'tasks.jsonl'}")
    print(f"  shards       : {out_dir / 'shards'} "
          f"({manifest['shard_size']}/shard, {manifest['n_shards']} shards)")


if __name__ == "__main__":
    main()
