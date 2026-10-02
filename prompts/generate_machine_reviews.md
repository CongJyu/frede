# Stage 1 — Generating machine-written review counterparts

This packet asks an external LLM to produce the **machine-written half** of a
paired review corpus. The human half is real Yelp reviews, already on disk.

## Why paired, and why three conditions

The shipped detector was trained with `1–2★ = fake`, so it learned sentiment,
not authorship. Mixing sentiment back in would reproduce that bug. Every
record here therefore **pins the star rating** and, for the rewrite condition,
**pins the content** — the only thing left that differs between the two halves
is *how the text is written*.

| `condition` | Source text given? | What it produces | Why it exists |
|---|---|---|---|
| `machine_rewrite` | yes | Same opinion and facts, rewritten in the model's natural voice | The core training signal. Content-matched to the human half, so sentiment cannot be a shortcut. |
| `machine_generate` | **no** (only rating + business type) | A review written from scratch | Matches how fake reviews are actually produced. Tests generalisation beyond rewriting. |
| `human_edit` | yes | The human review with spelling/grammar/punctuation fixed, nothing else | **Negative control.** If the detector fires on these, it is detecting *polish*, not machine authorship. |

`human_edit` is not optional. Without it there is no way to tell "this text was
written by a model" apart from "this text has no typos", and the resulting
detector would flag every literate reviewer.

> **This file is the design rationale. If you are the agent doing the
> generation, read `prompts/GENERATION_BRIEF.md` instead** — it is the
> self-contained operator brief with the output contract and procedure.

## How to run it

`data/gen_packet/shards/` holds the work, 50 records per file. The `prompt`
field of each record is the **complete, already-rendered user prompt** — send
it verbatim, do not re-template it. Records look like:

```jsonc
{
  "custom_id": "p000123_machine_rewrite",  // unique; echo this back
  "pair_id":   "p000123",
  "condition": "machine_rewrite",
  "split":     "train",                    // "train" | "dev"
  "stars":     4,                          // 1-5, must be reflected in output
  "source_text": "...",                    // the human review, or null
  "prompt":    "...rendered prompt..."     // send as-is
}
```

`data/gen_packet/system_prompt.txt` holds the system prompt to pair with it.

### Batch APIs

Pass `--vendor-files` to `make_gen_packet.py` to also emit pre-wrapped request
files (`requests.anthropic.jsonl`, `requests.openai.jsonl`). **Do not send
`temperature`/`top_p` to Opus 5, Sonnet 5, Fable 5, or Opus 4.8/4.7 — those
parameters were removed and return a 400.** Diversity comes from the thousands
of distinct source reviews, not from sampling knobs.

### Cross-generator evaluation

Run the whole packet through **at least two different models** and save each
run to its own file. Stage 2 trains on generator A and measures on generator B;
if performance collapses, the detector learned A's quirks rather than machine
authorship in general. That is the single most important number in this
project, and it cannot be computed from one generator.

## Output contract

One JSON object per line, nothing else:

```json
{"custom_id": "p000123_machine_rewrite", "text": "<the generated review>"}
```

Echo `custom_id` exactly. Results arrive in any order — Stage 2 keys on
`custom_id`, so ordering does not matter. If a request fails, omit it; do not
emit a placeholder.

## Quality bar

The data is only useful if the machine half is *realistic* modern LLM output.
Two failure modes ruin it, in opposite directions:

- **Caricature** ("Indulge in a culinary journey! ⭐⭐⭐⭐⭐") — trivially
  separable, and the detector learns a parody instead of a signal.
- **Over-imitation** — if the model is told to "write like a human", the
  machine half becomes stylometrically human and the task becomes impossible.

So the instruction in every prompt is: *write the way you naturally write*. Do
not ask the generating model to sound human, and do not ask it to sound like an
AI. Natural assistant output is exactly what a fake review produced by an LLM
looks like in the wild.

## Ingesting results

```bash
python -m scripts.ingest_gen_results \
    --results data/gen_results/<generator-tag> \
    --generator <generator-tag>
```

`--results` takes files or a whole shard-output directory. The ingest step
validates that each result maps to a known `custom_id`, that length and content
are plausible for its condition, and reports rejects under named reasons rather
than silently dropping them. `human_edit` outputs are additionally checked for
*content drift*: a control that rewrote the review instead of copy-editing it is
not a control.

`<generator-tag>` becomes the `generator` field, which is what the Stage 2
cross-generator split keys on — use a stable, filename-safe tag.
