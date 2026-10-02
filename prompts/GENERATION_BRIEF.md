# Generation brief — machine-written review corpus

You are generating the **machine-written half** of a paired review dataset used
to train a detector that tells machine-written text apart from human-written
text. The human half is real Yelp reviews and already exists. Your output is
the training signal for the machine class.

Read this whole file before starting. Everything you need is here; you do not
need to read any Python.

---

## 1. What you are given

```
data/gen_packet/            <- standard run: 60 shards x 50 records
  shards/                   <- YOUR INPUT
    shard_000.jsonl
    ...
  system_prompt.txt         <- the system prompt to use for every request
  tasks.jsonl               <- all 3000 records (do not work from this)
```

Two other shapes of the *same* packet exist — identical records, identical
`custom_id`s, only the sharding differs. Use whichever directory you were
pointed at; the shard filenames and everything below are the same:

| Directory | Shards | Records each | Use for |
|---|---|---|---|
| `data/gen_packet/` | 60 | 50 | a run from a generator already known to work |
| `data/gen_packet_small/` | 300 | 10 | a full run from a generator that struggled |
| `data/gen_packet_pilot/` | 3 | 10 | **deciding whether a generator can do this at all** |

Fewer records per shard means less to produce per work unit, which is the fix
for a run that came back short or repetitive. **If you have been pointed at the
pilot, do only those 3 shards and stop** — 30 records, then the check in §5. Do
not continue into a larger packet on your own initiative.

Each line of a shard is one review to generate:

```jsonc
{
  "custom_id": "p000042_machine_rewrite",  // echo this back exactly
  "pair_id":   "p000042",
  "condition": "machine_rewrite",          // "machine_rewrite" | "machine_generate" | "human_edit"
  "split":     "train",                    // ignore, bookkeeping
  "stars":     4,                          // 1-5. The output MUST reflect this rating.
  "source_text": "...",                    // the original human review, or null
  "business":  "a coffee shop",            // only for machine_generate, else null
  "prompt":    "..."                       // the COMPLETE instruction. See step 2.
}
```

## 2. The task, per record

For each record:

1. Send **`system_prompt.txt` verbatim as the system prompt**, and the
   record's **`prompt` field verbatim as the user message**.
2. **Do not re-template, summarise, shorten, or "improve" the `prompt` field.**
   It is already fully rendered and includes the `custom_id` you must echo.
   Adding your own preamble or restating the task counts as altering it.
3. Take the model's review text and write it out. That is the whole job.

The `prompt` field already tells the model which of the three conditions it is
doing; you do not need to branch on `condition` yourself. For reference:

| `condition` | What the prompt asks for | Why |
|---|---|---|
| `machine_rewrite` | Rewrite the given review with the same rating and the same facts, in the model's own voice | Content-matched to the human half, so sentiment cannot be used as a shortcut |
| `machine_generate` | Write a new review from scratch for a given business type and rating | Tests generalisation beyond rewriting |
| `human_edit` | Fix only spelling/grammar/punctuation in the given review, changing nothing else | **Negative control.** It must stay recognisably the same person's writing |

---

## 3. Output specification — this is the contract

Write **one output file per input shard**, at:

```
data/gen_results/<YOUR-TAG>/shard_000.jsonl
data/gen_results/<YOUR-TAG>/shard_001.jsonl
...
```

`<YOUR-TAG>` is a stable, filename-safe tag identifying the generating model,
lowercase with hyphens — for example `deepseek-v4f-cline` or `mimo-v2.5-opencode`.
Ask the operator for the exact tag if it is not already given to you. This tag
becomes the `generator` field that the train/test split keys on, so it must be
consistent across every file you write.

Each line is exactly one JSON object:

```json
{"custom_id": "p000042_machine_rewrite", "text": "The review text goes here..."}
```

Rules:

- **One output line per input line.** 50 in, 50 out.
- **`custom_id` must be copied byte-for-byte** from the input record. Do not
  modify it, recompute it, renumber it, or add a suffix.
- **`text` must be the bare review string** — the review and nothing else. No
  nested JSON, no markdown code fences, no ` ```json ` wrapper, no "Here is the
  review:" preamble, no trailing commentary.
- **Do not add any other fields.** No `condition`, no `stars`, no `model`, no
  token counts. Extra fields are not needed and may be discarded.
- Valid JSON on one line. Escape internal double quotes and newlines
  (`\n`) properly — prefer writing reviews without literal newlines.
- Records may appear in any order within the file. Order is not checked.
- **If a request fails or the output is unusable, omit that line.** Do not
  write a placeholder, a truncated review, or an error message in `text`.
  A missing line is recoverable; a placeholder is poison in the training set.
- **Every `text` must be generated fresh from its own record.** Do not reuse,
  lightly reword, or fall back on a stock review when the source is long or the
  going gets slow. A run that repeats the same review across many records looks
  complete but trains the detector on a handful of strings — it is worse than an
  incomplete run, because the damage is silent. If you find yourself producing
  near-duplicates, stop and report it instead of finishing the shard.
- **`machine_rewrite` means rewrite, not word substitution.** Swapping
  "restaurant"→"venue" and "food"→"cuisine" while leaving every sentence intact
  is not a rewrite, and it is rejected automatically. Rebuild each sentence in
  your own words; the source's *meaning* must survive, its *phrasing* must not.
  Do not join or split words, and do not introduce typos.
- **Preserve the source's paragraph breaks.** The source may have blank lines
  separating paragraphs. `machine_rewrite` and `human_edit` must keep the same
  breaks — do not collapse a multi-paragraph review into one block, and do not
  invent new paragraphs. (`machine_generate` decides its own.)
- **Sources over ~500 words need care.** Some are long enough that generation
  gets cut off partway. Write those out completely; a half-finished review is
  discarded.

A file containing zero valid lines is worse than no file at all — if a shard
fails entirely, delete the output file rather than leaving it empty.

---

## 4. Procedure

**If this is a pilot run, stop after the shards you were given** and go straight
to the check in step 5. A run can be 100% complete on every shard and still be
unusable — one past run scored full coverage with 41% distinct outputs and
rewrites that never read their source. The check takes minutes and is the only
thing that distinguishes those cases. If it does not come back `USABLE`, stop
and report; do not grind through more shards.

Work **one shard at a time**. After each shard:

1. Count the lines in your output file. It must equal the input shard's line
   count minus any lines you deliberately omitted.
2. Spot-check one `human_edit` record: the output should be nearly identical to
   its `source_text` (only spelling/grammar/punctuation changed). If it reads
   like a rewrite, the prompt was not followed — redo that shard.
3. Move to the next shard.

**Resumption:** before starting a shard, check whether
`data/gen_results/<YOUR-TAG>/shard_NNN.jsonl` already exists and is complete.
If it is, skip that shard and go to the next one. This makes the run restartable
— never redo completed shards, and never rewrite a file you already finished.

Validate your work at any point by running the health check:

```bash
.venv/bin/python -m scripts.check_pairs \
    --results data/gen_results/<YOUR-TAG> \
    --generator <YOUR-TAG>
```

This is the command that decides whether the run is usable. It reports, per
condition: whether outputs are distinct (template collapse), whether rewrites
are actually bound to their source, whether the source's opinion survived, and
whether anything is truncated. It ends with a `USABLE` / `NOT USABLE` verdict
and names every problem it found.

`.venv/bin/python -m scripts.ingest_gen_results --results <dir> --generator <tag>`
additionally reports coverage against the packet and per-record reject reasons;
use it to confirm nothing was dropped.

Run the health check **after your pilot shards and again at the end**, and
report both the verdict and the coverage number.

---

## 5. Quality bar — the part that is easy to get wrong

The dataset is only useful if your machine-written reviews are **realistic
modern LLM output**. Two opposite failure modes destroy it:

- **Caricature.** "Indulge in a culinary journey! ⭐⭐⭐⭐⭐" — a parody of AI
  writing. A detector trained on this learns to spot the parody, not machine
  authorship. Emoji, star symbols, marketing superlatives stacked three deep,
  and "As an AI language model..." are all failures.
- **Over-imitation.** If the reviews are written to sound like a human typing
  on their phone — typos, fragments, slang — the machine half becomes
  stylometrically human and the task becomes impossible.

The prompts already instruct the model to *write the way it naturally writes*.
Your job is to not undermine that: do not add instructions telling it to "sound
human", do not add instructions telling it to "sound like an AI", and do not
hand-write reviews yourself. **The review text must come from the model under
the given prompt.** If you write them yourself, the dataset measures your
writing style, not the model's, and the whole exercise is void.

Other things that will get records rejected downstream:

- Meta-commentary ("As an AI, I cannot...", "I'm sorry, but...")
- Empty or near-empty reviews
- For `human_edit`: content drift — a copy-edit that turned into a rewrite
- For `machine_rewrite`: output that just copies the source (near-identical text)
- Length wildly off: `human_edit` must stay within ~±20% of the source;
  `machine_rewrite` roughly matches its source; `machine_generate` is 60–160 words

---

## 6. Report back

When finished, report:

1. Which shards you completed (and any you skipped or omitted, with why).
2. The coverage number from the ingest command in step 4.
3. Anything about the prompt or the records that seemed broken or ambiguous.
   Finding a problem with the packet is useful output — do not paper over it.
