# Plan — from here to a defensible result

Written 2026-10-02. Status of the project at this point, what is blocked and on
what, and the order to do things in. Facts below were verified against the
working tree, not recalled.

## Where things stand

The pipeline is rebuilt end to end and runs:

- **Stage 0** — zero-shot signals exist and preprocessing destroys ~0.19 of them.
- **Stage 1** — packet, generator brief, health checks, ingest gates.
- **Stage 2** — detector on genuine MiMo data: CV AUROC **0.9595 ± 0.0084**,
  held-out **0.9572**, TPR@1%FPR **0.655**, probe FPRs 0.49% / 0.75%.
- **Stage 3** — serving layer rebuilt: GPT-2 surprisal + SHAP, 60–300 ms per
  request, reason codes, Analyzer page.

One generation run turned out to be synthetic and was caught
(`gen_reviews.py`); the gates that catch that class of failure now exist and
have been verified against both known-bad runs.

## The cross-generator question is already answered (partly)

The detector's top features are stylometric — `avg_word_len` (dAUC +0.077),
`has_em_dash`, `excl_per_sent`, `contraction_rate` — and stylometric alone
reaches 0.9430 versus surprisal alone at 0.8189, the *inverse* of Stage 0's
finding against weak generators. The worry was that the detector keys on MiMo's
house style rather than on authorship.

**It is measurable now, without a second generation run.** Evaluation needs only
the text, not alignment to the packet — and genuine output from other models is
already on disk. Scoring it:

| source | n | median P(machine) | flagged |
|---|---|---|---|
| MiMo rerun (trained on this) | 400 | 0.996 | 87.2% |
| MiMo first run — same model, different session | 400 | 0.997 | 77.5% |
| **DeepSeek V4F — different model** | 400 | **0.942** | **40.2%** |
| DeepSeek rewrites | 400 | 0.313 | 12.5% |
| human, held-out Yelp | 297 | 0.007 | 0.0% |

Reading: **there is real cross-model transfer, and it decays.** Same model in a
different session barely moves (0.997 / 78%), so the detector is not keyed to one
session's quirks. A different model drops sharply (0.942 / 40%) but stays far
from human. The detector is neither a MiMo-style detector nor a universal
machine-text detector; it sits between, and the distance is now quantified.

**Correction to an earlier reading.** Scoring GPT-2 and Pythia output gives
median P = 0.004, flagged 0%. That looked like zero transfer but is not
informative: those are base LMs whose continuation does not resemble a *review*
at all. The detector declining to call incoherent rambling a model-written
review is correct behaviour for a task about reviews. Only a generator that
writes actual reviews probes the right thing.

## The one thing still blocked

**Training on two generators.** The evaluation probe above does not require it,
but the training-side question does: does training on generator A and B improve
generalisation to C? That needs a genuine second run.

DeepSeek V4F (via Cline) remains the candidate — its output is genuine (3%
script reproduction) and it rewrote rather than substituted (verbatim 0.60
median). It failed only by running against an earlier packet.

```bash
# hand prompts/GENERATION_BRIEF.md to the generator, point it at the frozen
# packet (data/gen_packet/, 60 shards x 50), output to a new directory, then:
make check-pairs RESULTS=data/gen_results/<tag> GENERATOR=<tag>
make ingest-gen  RESULTS=data/gen_results/<tag> GENERATOR=<tag>
make stage2 GENERATORS="mimo-v2.5-rerun <tag>"
```

`make stage2` needs no code change — the per-generator table grows a row.

**Verdict: skipping the re-run is reasonable.** The headline claim is already
narrowed and measured; a second training generator would strengthen it, not
rescue it. It should not block Phase 2.

## Order of work

**Phase 1 — SKIPPED (re-run), DONE (measurement).** The cross-generator
evaluation is in hand and reported above. The re-run would only add the
training-side question, which is a strengthening, not a prerequisite. Do not
hold other work behind it.

**Phase 2 — Stage 4: HITL and Examples.  DONE.**
Both pages now run on the current detector. The HITL question is
human-written vs machine-written with authorship as ground truth; Examples shows
one case per confusion-matrix quadrant, with the correct calls at their most
confident and the errors at the decision boundary. `make stage4` rebuilds both.

The current inconsistency is real and visible to a user — the Analyzer answers
"human or machine?" while HITL and Examples still answer "fake or real?" with the
old sentiment model. Work:

- Rewrite `training/examples.py` for the new detector: TP/TN/FP/FN with SHAP
  feature attributions and token-predictability highlighting.
- Rewrite `training/hitl_pool.py` likewise.
- **Change the HITL question** from "is this review fake?" to "was this written
  by a person or a model?". The two-pass design (raw, then with XAI) is the
  study's actual research question and is unaffected; only the question and the
  ground truth change.
- Note the human side is now genuinely human-authored (real Yelp), so HITL
  accuracy is meaningful in a way it was not under the star proxy.

**Phase 3 — `machine_rewrite` quality.** 371 of 999 passed; the rest were
synonym substitution. The survivors are the better end of a poor distribution,
so the 0.8978 for that condition is optimistic and the pooled 0.9572 is weighted
toward the easier `machine_generate`. Options: re-run that condition alone with
a sharper prompt (200 shards), or keep the 371 and state the bias. Decide after
Phase 1 is skipped, so this is now a re-run-or-accept call on its own.

**Phase 4 — housekeeping.  MOSTLY DONE.**

- DONE: the three legacy modules are removed and the app no longer loads the
  sentiment model at startup. `training/train.py` is decoupled from the Stage 4
  builders, which it could no longer call.
- DONE: both Makefile and `training/stage2.py` defaulted to `mimo-v2.5-full` —
  the synthetic run. Anyone running either without arguments would have trained
  on template text. Now defaults to `mimo-v2.5-rerun`.
- REMAINING: README's opening still describes the old sentiment pipeline as the
  headline. It should lead with machine-generated detection.
- REMAINING: everything is uncommitted (2 commits in the repo).
  `scripts/gen_reviews.py` is untracked and is the provenance record for a
  rejected dataset — commit it with a message saying so, do not delete it.

## Open decisions

1. ~~How much does the single-generator caveat limit what can be claimed?~~
   **Answered.** Real transfer that decays: 78% flagged on the same model in a
   different session, 40% on a different model, 0% on humans. The claim is
   "detects model-written reviews, with transfer that weakens across models" —
   and the decay curve is itself a result worth reporting.
2. **Is `machine_rewrite` worth re-running?** It is the more realistic condition
   and currently the weakest (371/999 passed; the survivors are the better end of
   a poor distribution). Re-running it alone costs ~200 shards. Decide after
   Phase 2 — HITL does not depend on it.
3. **What is the deliverable?** The project has a coherent story — the star-proxy
   reframing, the escape-artifact trap, the synthetic-run incident, the gates
   built in response, and results on genuine data. That is a paper or an internal
   writeup; it is not yet written down anywhere but the README.

## Things that are done and should not be redone

- Stage 0's conclusions (real human text, real reference models).
- The packet — it is frozen and guarded; regenerating it invalidates any run
  keyed to it.
- The gates. Both have been verified against the failures they exist to catch.
