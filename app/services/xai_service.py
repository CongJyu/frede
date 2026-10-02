"""Per-request analysis for the machine-generated-review detector.

Two levels of explanation, both exact rather than sampled:

- **Token level** — every token's probability under the reference LM. This is
  not a surrogate: it is the same quantity the detector reads, so a highlighted
  token is directly part of the evidence, not an approximation of it.
- **Feature level** — SHAP values from the gradient-boosted classifier's own
  trees, computed exactly in ~30 ms.

The previous pipeline ran LIME with 150 samples per request and Integrated
Gradients over a transformer, costing 3–8 s and explaining a surrogate fit
around the model. Neither is needed here: the model is small and its features
are already interpretable, so the whole request lands well under a second.

Token highlighting marks **high-probability** tokens — text the reference model
found predictable is the machine-authored signature, so red means "predictable".
Note this is the opposite convention to the removed sentiment pipeline, where
red meant "pushes toward fake".
"""
from __future__ import annotations

import html
import math
import time

import numpy as np

from .. import config
from ..reason_codes import generate_reason_codes, reason_code_items
from .detector_service import get_detector_service

# Tokens at or above this probability are predictable enough to be worth
# marking; below it they are ordinary words carrying ordinary information.
_HIGHLIGHT_FLOOR = 0.30


def _readable(token: str) -> str:
    """GPT-2 byte-level tokens to display text (`Ġ` is a leading space)."""
    return token.replace("Ġ", " ").replace("Ċ", "\n")


def build_highlighted_html(tokens: list[str], logprobs: list[float],
                           max_tokens: int = 400) -> str:
    """Colour each token by how predictable the reference model found it.

    `logprobs[i]` is log p(token_i | token_<i). Red intensity tracks the
    probability, so the strongest marks are the words the model would itself
    have chosen — the visible shape of machine authorship.
    """
    if not tokens:
        return '<p style="color:#888">Not enough text to score.</p>'

    parts = [
        '<div style="font-size:16px; line-height:2.2; '
        'font-family:Georgia,serif; padding:10px; white-space:pre-wrap;">'
    ]
    for token, lp in list(zip(tokens, logprobs))[:max_tokens]:
        text = html.escape(_readable(token))
        if not text:
            continue
        prob = math.exp(min(0.0, lp))
        if prob >= _HIGHLIGHT_FLOOR:
            intensity = min((prob - _HIGHLIGHT_FLOOR) / (1 - _HIGHLIGHT_FLOOR), 1.0)
            alpha = 0.15 + 0.7 * intensity
            parts.append(
                f'<span style="background:rgba(200,40,40,{alpha:.3f}); '
                f'border-radius:3px; padding:1px 2px;" '
                f'title="p={prob:.3f} — the reference model expected this word">'
                f"{text}</span>"
            )
        else:
            parts.append(text)
    parts.append("</div>")
    parts.append(
        '<p style="font-size:12px; color:#666; margin-top:8px;">'
        "Red = the reference language model found this word predictable. "
        "Machine-written text scores high across the whole review; "
        "hover a token for its probability.</p>"
    )
    return "".join(parts)


def _feature_table(result: dict, ds) -> list[dict]:
    """Features sorted by how much each moved this decision."""
    rows = []
    for name, value in result["features"].items():
        shap_value = result["shap"].get(name, 0.0)
        pct = ds.human_percentile(name, value)
        rows.append({
            "name": name,
            "value": round(float(value), 6),
            "human_percentile": None if pct is None else round(pct, 1),
            "shap": round(float(shap_value), 5),
            # Positive SHAP pushes toward machine; the UI colours on this.
            "direction": "machine" if shap_value > 0 else "human",
        })
    rows.sort(key=lambda r: -abs(r["shap"]))
    return rows


def _scope_note(generators: list[str]) -> str:
    names = ", ".join(generators) if generators else "an unknown generator"
    if len(generators) <= 1:
        return (
            f"Trained on machine-written reviews from a single generator "
            f"({names}). Scores are trustworthy against that generator; "
            f"cross-generator generalisation is not yet measured."
        )
    return f"Trained on machine-written reviews from: {names}."


def run_analysis(review_text: str) -> dict:
    start = time.time()
    ds = get_detector_service()

    if not review_text or not review_text.strip():
        return {
            "prediction": "N/A", "predicted": -1, "machine_prob": 0.0,
            "confidence": 0.0, "flagged": False, "threshold": ds.threshold,
            "features": [], "tokens": [], "logprobs": [],
            "highlighted_html": "<p>Please enter a review.</p>",
            "reason_codes": [], "summary": "No text provided.",
            "elapsed_ms": 0, "scope_note": _scope_note(ds.generators),
        }

    result = ds.score(review_text)
    prob = result["prob"]
    # The verdict is taken at the threshold calibrated to hold false positives
    # on real reviews at 1%, not at 0.5. A 0.5 boundary carries no FPR guarantee
    # — on this detector it implies roughly an order of magnitude more false
    # accusations — so using it would state a verdict the model never promised.
    predicted = int(prob >= ds.threshold)

    percentiles = {c: ds.human_percentile(c, v) for c, v in result["features"].items()}
    rc = generate_reason_codes(
        features=result["features"],
        percentiles=percentiles,
        shap=result["shap"],
        machine_prob=prob,
        predicted=predicted,
        cut=config.RC_PERCENTILE_CUT,
        threshold=ds.threshold,
    )

    return {
        "prediction": "MACHINE" if predicted == 1 else "HUMAN",
        "predicted": predicted,
        "machine_prob": round(prob, 5),
        "confidence": round(prob if predicted == 1 else 1 - prob, 5),
        # Identical to `predicted` by construction; kept as a separate field
        # because the UI states it in terms of the operating guarantee.
        "flagged": predicted == 1,
        "threshold": round(ds.threshold, 5),
        "features": _feature_table(result, ds),
        "tokens": result["tokens"],
        "logprobs": [round(v, 5) for v in result["logprobs"]],
        "highlighted_html": build_highlighted_html(result["tokens"], result["logprobs"]),
        "reason_codes": reason_code_items(rc),
        "summary": rc.summary,
        "elapsed_ms": int((time.time() - start) * 1000),
        "scope_note": _scope_note(ds.generators),
    }


__all__ = ["run_analysis", "build_highlighted_html", "get_detector_service"]
