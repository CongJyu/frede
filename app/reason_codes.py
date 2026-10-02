"""Reason codes for machine-generated review detection.

The previous taxonomy described a sentiment classifier: promotional language,
urgency phrases, extreme polarity. Those codes were artifacts of reading 1-2
star reviews as "fake", and they said nothing about authorship. This taxonomy
replaces them; the old modules were removed once HITL and Examples moved to the
current detector.

Each code is anchored to a feature whose separation between human and
machine-written reviews was **measured** in Stage 2, and each triggers only when
the value falls outside the range real reviewers occupy — expressed as a
percentile of the human training distribution, never as an absolute number. A
code therefore means "this text behaves unlike the real reviews we learned
from", which is a claim the data can actually support.

Direction and effect size come from `data/stage2_report.json`
(`single_feature_auroc`); the percentiles come from the reference profile saved
alongside the model. Changing either invalidates the taxonomy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# `trigger_percentile` is the cut in the HUMAN distribution: "high" fires above
# it, "low" below it. The outer 5% is deliberately wide so a code marks a real
# outlier rather than ordinary variation between reviewers.
REASON_CODES: dict[str, dict] = {
    "RC-01": {
        "label": "Excessive Predictability",
        "description": (
            "The text is more predictable to a reference language model than "
            "95% of real reviews. Machine-written text tends to sit near the "
            "model's own most-likely continuation."
        ),
        "features": {"surp_lp_mean": "high", "surp_ppl": "low"},
        "trigger_percentile": 95,
        "source": "GPT-2 surprisal",
    },
    "RC-02": {
        "label": "Uniform Sentence Rhythm",
        "description": (
            "Sentence lengths are unusually uniform. Real reviewers mix very "
            "short and very long sentences; machine text keeps a steady pace."
        ),
        "features": {"style_sent_len_std": "low", "style_sent_len_cv": "low"},
        "trigger_percentile": 95,
        "source": "stylometric",
    },
    "RC-03": {
        "label": "Out-of-Distribution Vocabulary",
        "description": (
            "Average word length sits above 95% of real reviews — the "
            "vocabulary is more formal or elaborate than customers typically use."
        ),
        "features": {"style_avg_word_len": "high"},
        "trigger_percentile": 95,
        "source": "stylometric",
    },
    "RC-04": {
        "label": "Flat Information Profile",
        "description": (
            "Token-by-token surprisal barely varies. Human reviews are bursty — "
            "a surprising detail, then a predictable clause — while machine text "
            "spreads information evenly."
        ),
        "features": {"surp_lp_std": "low", "surp_mean_entropy": "low"},
        "trigger_percentile": 95,
        "source": "GPT-2 surprisal",
    },
    "RC-05": {
        "label": "High Next-Token Agreement",
        "description": (
            "The reference model's single best guess was the actual next word "
            "far more often than in real reviews, a signature of text generated "
            "by sampling from such a model."
        ),
        "features": {"surp_frac_top1": "high"},
        "trigger_percentile": 95,
        "source": "GPT-2 surprisal",
    },
    "RC-06": {
        "label": "Absence of Concrete Specifics",
        "description": (
            "Fewer numerals than 95% of real reviews. Customers cite prices, "
            "waits, dates and counts; this text stays general."
        ),
        "features": {"style_digit_per_sent": "low"},
        "trigger_percentile": 95,
        "source": "stylometric",
    },
}


@dataclass
class ReviewExplanation:
    text: str
    predicted: int
    machine_prob: float
    reason_codes: list[str] = field(default_factory=list)
    evidence: dict[str, list] = field(default_factory=dict)
    summary: str = ""

    def __str__(self) -> str:
        sep = "─" * 60
        lines = [
            sep,
            f"Text     : {self.text[:200]}{'...' if len(self.text) > 200 else ''}",
            f"Predicted: {'MACHINE' if self.predicted == 1 else 'HUMAN'} "
            f"(P(machine)={self.machine_prob:.3f})",
        ]
        if self.reason_codes:
            lines.append("\nReason Codes:")
            for code in self.reason_codes:
                meta = REASON_CODES[code]
                lines.append(f"  [{code}] {meta['label']}")
                lines.append(f"         {meta['description']}")
                if code in self.evidence:
                    lines.append(f"         Evidence: {self.evidence[code]}")
        else:
            lines.append("No strong machine-authorship signals detected.")
        lines.append(f"\nSummary : {self.summary}")
        lines.append(sep)
        return "\n".join(lines)


def _fmt(feature: str, value: float, pct: float | None, shap: float | None) -> str:
    scale = 4 if abs(value) < 10 else 2
    bits = f"{feature}={value:.{scale}f}"
    if pct is not None:
        # Clamped at the ends; np.interp cannot see beyond the outermost
        # reference percentile, so "≤1st" / "≥99th" is the honest phrasing.
        if pct <= 1:
            bits += " (≤1st pct of real reviews)"
        elif pct >= 99:
            bits += " (≥99th pct of real reviews)"
        else:
            bits += f" ({pct:.0f}th pct of real reviews)"
    if shap is not None:
        bits += f" [shap {shap:+.3f}]"
    return bits


def generate_reason_codes(
    *,
    features: dict[str, float],
    percentiles: dict[str, float | None],
    shap: dict[str, float] | None = None,
    machine_prob: float,
    predicted: int,
    cut: float = 5.0,
    threshold: float | None = None,
) -> ReviewExplanation:
    """Map feature values to reason codes.

    A code fires when any of its features falls outside the human distribution,
    in the direction machine text is known to deviate. Evidence lists the
    triggering features ordered by their SHAP contribution, so the explanation
    leads with what actually moved this decision.
    """
    shap = shap or {}
    codes: list[str] = []
    evidence: dict[str, list[str]] = {}

    for code, meta in REASON_CODES.items():
        hits: list[tuple[float, str]] = []
        for feature, direction in meta["features"].items():
            value = features.get(feature)
            pct = percentiles.get(feature)
            if value is None or pct is None:
                continue
            outside = pct >= (100 - cut) if direction == "high" else pct <= cut
            if outside:
                hits.append((abs(shap.get(feature, 0.0)),
                             _fmt(feature, value, pct, shap.get(feature))))
        if hits:
            hits.sort(key=lambda h: -h[0])
            codes.append(code)
            evidence[code] = [text for _, text in hits]

    # A reason code reports a fact — this feature really is outside the range
    # real reviews occupy — which is independent of which way the verdict went.
    # Both directions therefore have to be described honestly, including the
    # case where machine-typical signals are present but outweighed.
    labels = [REASON_CODES[c]["label"] for c in codes]
    if predicted == 1:
        summary = (
            f"Flagged as likely MACHINE-WRITTEN (P={machine_prob:.1%}). "
            f"Signals outside the range of real reviews: {'; '.join(labels)}."
            if codes else
            f"Flagged MACHINE-WRITTEN (P={machine_prob:.1%}) on the combined "
            f"feature pattern, but no single signal falls outside the range real "
            f"reviews occupy — treat this as weak evidence."
        )
    elif threshold is not None and machine_prob >= 0.5:
        # Between the 0.5 boundary and the calibrated cut: the model does lean
        # machine, but not by enough to meet the operating guarantee. Calling
        # this "human" without saying so would misreport what the model saw.
        #
        # The wording tracks the magnitude. "Leans slightly" is wrong at 92%,
        # and a hedge that understates the score is the one a reader is most
        # likely to act on.
        lean = "slightly" if machine_prob < 0.75 else "strongly"
        summary = (
            f"Below the flagging threshold (P(machine)={machine_prob:.1%} vs "
            f"{threshold:.1%} required to hold false positives on real reviews at "
            f"1%). Leans {lean} machine but not enough to flag."
            + (f" Signals present: {'; '.join(labels)}." if codes else "")
        )
    else:
        summary = (
            f"Classified HUMAN-WRITTEN (P(machine)={machine_prob:.1%}). "
            f"{len(codes)} machine-typical signal(s) are present but outweighed by "
            f"the rest of the profile: {'; '.join(labels)}."
            if codes else
            f"No machine-authorship signals outside the range of real reviews. "
            f"Classified HUMAN-WRITTEN (P(machine)={machine_prob:.1%})."
        )

    return ReviewExplanation(
        text="",
        predicted=predicted,
        machine_prob=machine_prob,
        reason_codes=codes,
        evidence=evidence,
        summary=summary,
    )


def reason_code_items(rc: ReviewExplanation) -> list[dict]:
    """Flatten reason codes into JSON-friendly dicts.

    `aligns` reports whether the signal points the same way as the verdict. The
    UI greys out codes that contradict it, so a reader is never shown a
    machine-typical signal as if it justified a HUMAN call.
    """
    return [
        {
            "code": code,
            "label": REASON_CODES[code]["label"],
            "description": REASON_CODES[code]["description"],
            "source": REASON_CODES[code]["source"],
            "aligns": rc.predicted == 1,
            "evidence": rc.evidence.get(code, []),
        }
        for code in rc.reason_codes
    ]
