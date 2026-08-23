"""Reason-code engine — maps attributions to human-readable explanation codes.

Port of notebook Phase 3 - Part 4 (cell 33). Note the original cell never
appends its generated explanations to the output list (a bug that leaves its
CSV empty); we compute reason codes per request instead, so the bug is not
reproduced here.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Reason Code Taxonomy
REASON_CODES: dict[str, dict] = {
    "RC-01": {
        "label": "Excessive Promotional Language",
        "description": (
            "Review contains an unusually high density of superlative or "
            "marketing-style words that inflate sentiment artificially."
        ),
        "trigger_words": {
            "amazing", "incredible", "fantastic", "perfect", "best", "awesome",
            "outstanding", "phenomenal", "superb", "excellent", "wonderful",
            "brilliant", "magnificent", "unbelievable", "flawless",
        },
    },
    "RC-02": {
        "label": "Generic / Template-like Content",
        "description": (
            "Review uses common phrases found across many fake reviews and "
            "lacks specific, personal detail."
        ),
        "trigger_words": {
            "highly recommend", "must try", "worth every penny", "five stars",
            "ten out of ten", "do not hesitate", "rush", "hurry",
            "everyone should", "trust me",
        },
    },
    "RC-03": {
        "label": "Extreme Sentiment Polarity",
        "description": (
            "Review's predicted confidence is very high (>0.85), indicating "
            "an atypically one-sided opinion."
        ),
        "trigger_condition": "high_confidence",
    },
    "RC-04": {
        "label": "Suspicious Urgency / Action-Prompt Language",
        "description": (
            "Review pushes the reader to take immediate action, a pattern "
            "associated with incentivised/fake reviews."
        ),
        "trigger_words": {
            "go now", "visit today", "book immediately", "hurry up",
            "limited time", "dont wait", "you wont regret", "call now",
            "order now", "grab",
        },
    },
    "RC-05": {
        "label": "High XAI Attribution to Non-Descriptive Tokens",
        "description": (
            "SHAP/LIME/IG flags tokens that are not descriptive of the actual "
            "experience, suggesting manufactured content."
        ),
        "trigger_condition": "xai_non_descriptive",
    },
}

DESCRIPTIVE_TOKENS = {
    "food", "service", "staff", "price", "taste", "flavor", "portion", "menu",
    "table", "wait", "ambiance", "atmosphere", "location", "fresh", "cooked",
    "served", "ordered", "meal", "dish", "drink",
}


@dataclass
class ReviewExplanation:
    text: str
    true_label: int
    predicted: int
    fake_prob: float
    reason_codes: list[str] = field(default_factory=list)
    evidence: dict[str, list] = field(default_factory=dict)
    summary: str = ""

    def __str__(self):
        sep = "─" * 60
        lines = [
            sep,
            f"Text     : {self.text[:200]}{'...' if len(self.text) > 200 else ''}",
            f"Predicted: {'FAKE' if self.predicted == 1 else 'REAL'} "
            f"(P(Fake)={self.fake_prob:.3f})  |  "
            f"True: {'FAKE' if self.true_label == 1 else 'REAL'}",
        ]
        if self.reason_codes:
            lines.append("\nReason Codes:")
            for rc in self.reason_codes:
                meta = REASON_CODES[rc]
                lines.append(f"  [{rc}] {meta['label']}")
                lines.append(f"         {meta['description']}")
                if rc in self.evidence:
                    lines.append(f"         Evidence: {self.evidence[rc]}")
        else:
            lines.append("No strong fake-review signals detected.")
        lines.append(f"\nSummary : {self.summary}")
        lines.append(sep)
        return "\n".join(lines)


def generate_reason_codes(
    text: str,
    true_label: int,
    predicted: int,
    fake_prob: float,
    lime_exp=None,  # lime Explanation object (or None)
    shap_tokens: list[str] | None = None,
    shap_attrs: np.ndarray | None = None,
    ig_tokens: list[str] | None = None,
    ig_attrs: np.ndarray | None = None,
    shap_threshold: float = 0.05,
    ig_threshold: float = 0.10,
    confidence_threshold: float = 0.85,
) -> ReviewExplanation:
    reason_codes: list[str] = []
    evidence: dict[str, list] = {}
    lower_text = text.lower()

    # RC-01: Promotional language
    promo_hits = [
        w for w in REASON_CODES["RC-01"]["trigger_words"] if w in lower_text.split()
    ]
    if len(promo_hits) >= 2:
        reason_codes.append("RC-01")
        evidence["RC-01"] = promo_hits

    # RC-02: Template / generic phrases
    generic_hits = [
        p for p in REASON_CODES["RC-02"]["trigger_words"] if p in lower_text
    ]
    if generic_hits:
        reason_codes.append("RC-02")
        evidence["RC-02"] = generic_hits

    # RC-03: Extreme confidence
    if fake_prob >= confidence_threshold:
        reason_codes.append("RC-03")
        evidence["RC-03"] = [f"P(Fake)={fake_prob:.3f}"]

    # RC-04: Urgency language
    urgency_hits = [
        p for p in REASON_CODES["RC-04"]["trigger_words"] if p in lower_text
    ]
    if urgency_hits:
        reason_codes.append("RC-04")
        evidence["RC-04"] = urgency_hits

    # RC-05: XAI attribution to non-descriptive tokens
    flagged_non_descriptive: list[str] = []

    # From LIME
    if lime_exp is not None:
        lime_features = lime_exp.as_list(label=1)
        for word, weight in lime_features:
            clean = re.sub(r"[^a-z]", "", word.lower())
            if weight > 0.05 and clean and clean not in DESCRIPTIVE_TOKENS:
                flagged_non_descriptive.append(f"LIME:{clean}({weight:+.3f})")

    # From SHAP
    if shap_tokens is not None and shap_attrs is not None:
        top_indices = np.argsort(shap_attrs)[::-1][:10]
        for i in top_indices:
            tok = shap_tokens[i].replace("Ġ", "").replace("▁", "").lower()
            val = float(shap_attrs[i])
            if (
                val > shap_threshold
                and tok
                and tok not in DESCRIPTIVE_TOKENS
                and tok not in ("<s>", "</s>", "<pad>")
            ):
                flagged_non_descriptive.append(f"SHAP:{tok}({val:+.3f})")

    # From IG
    if ig_tokens is not None and ig_attrs is not None:
        top_indices = np.argsort(ig_attrs)[::-1][:10]
        for i in top_indices:
            tok = ig_tokens[i].replace("Ġ", "").replace("▁", "").lower()
            val = float(ig_attrs[i])
            if (
                val > ig_threshold
                and tok
                and tok not in DESCRIPTIVE_TOKENS
                and tok not in ("<s>", "</s>", "<pad>")
            ):
                flagged_non_descriptive.append(f"IG:{tok}({val:+.3f})")

    if len(flagged_non_descriptive) >= 3:
        reason_codes.append("RC-05")
        evidence["RC-05"] = flagged_non_descriptive[:8]

    # Human-readable summary
    if predicted == 1:
        if reason_codes:
            labels = [REASON_CODES[rc]["label"] for rc in reason_codes]
            summary = (
                f"This review was flagged as FAKE (confidence {fake_prob:.1%}). "
                f"Key signals: {'; '.join(labels)}."
            )
        else:
            summary = (
                f"Review predicted FAKE (confidence {fake_prob:.1%}) by the model, "
                f"but no explicit rule-based signals were matched. Prediction relies "
                f"primarily on learned model patterns."
            )
    else:
        summary = (
            f"No strong deception signals detected. Review predicted REAL "
            f"(confidence {1 - fake_prob:.1%})."
        )

    return ReviewExplanation(
        text=text,
        true_label=true_label,
        predicted=predicted,
        fake_prob=fake_prob,
        reason_codes=list(dict.fromkeys(reason_codes)),
        evidence=evidence,
        summary=summary,
    )


def reason_code_items(rc: ReviewExplanation) -> list[dict]:
    """Flatten reason codes into JSON-friendly {code, label, description, evidence}."""
    items = []
    for code in rc.reason_codes:
        meta = REASON_CODES[code]
        items.append(
            {
                "code": code,
                "label": meta["label"],
                "description": meta["description"],
                "evidence": rc.evidence.get(code, []),
            }
        )
    return items


def rc_frequency(predictions: list[ReviewExplanation]) -> dict[str, int]:
    counter: defaultdict[str, int] = defaultdict(int)
    for e in predictions:
        for rc in e.reason_codes:
            counter[rc] += 1
    return dict(counter)
