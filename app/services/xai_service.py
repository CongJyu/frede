"""Per-request analysis: predict + LIME + Integrated Gradients + Reason Codes.

This is the port of notebook Phase 4 Part 1 (`predict_and_explain`), with the
highlighted HTML and reason codes produced server-side and returned as JSON.
SHAP is intentionally NOT computed here (minutes per review on CPU); precomputed
SHAP lives on the Examples page instead.
"""
from __future__ import annotations

import html
import re
import time

import numpy as np
import torch
from captum.attr import IntegratedGradients

from .. import config
from ..preprocess import preprocess
from ..reason_codes import REASON_CODES, generate_reason_codes, reason_code_items
from .model_service import get_model_service


def _get_embedding_layer(model):
    """Return the word-embedding module, regardless of the base architecture."""
    for name in ("distilbert", "roberta", "bert"):
        sub = getattr(model, name, None)
        if sub is not None:
            return sub.embeddings
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            return mod
    raise RuntimeError("Could not locate an embedding layer on the model")


def get_ig_attributions(
    text: str,
    target_class: int = 1,
    n_steps: int = config.IG_N_STEPS,
    *,
    model=None,
    tokenizer=None,
    device=None,
) -> tuple[list[str], np.ndarray]:
    """Token-level Integrated Gradients via Captum, L2-normalised.

    Defaults to the process-wide ModelService singleton; pass `model` /
    `tokenizer` / `device` explicitly to reuse an in-memory (freshly trained)
    model instead. Returns (tokens, attributions) where positive attribution
    pushes toward the target class (Fake).

    Uses `captum.attr.IntegratedGradients` over explicitly-computed word
    embeddings (equivalent to LayerIntegratedGradients on the embedding layer,
    but avoiding a captum 0.9 `_extract_device` regression that crashes when the
    embedding hook receives a None `input_ids`).
    """
    if model is None:
        ms = get_model_service()
        model, tokenizer, device = ms.model, ms.tokenizer, ms.device
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    baseline_ids = torch.zeros_like(input_ids)  # all [PAD]

    embedding_layer = _get_embedding_layer(model)
    input_embeds = embedding_layer(input_ids)
    baseline_embeds = embedding_layer(baseline_ids)

    def forward_func(embeds, attention_mask):
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        return torch.softmax(outputs.logits.float(), dim=-1)[:, target_class]

    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum over embedding dim to (seq_len,), then drop padding tokens.
    attr = attributions.squeeze(0).sum(dim=-1).detach().cpu().numpy()
    mask = attention_mask.squeeze(0).cpu().numpy().astype(bool)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    tokens = [t for t, m in zip(tokens, mask) if m]
    attr = attr[mask]

    norm = np.linalg.norm(attr)
    attr = attr / (norm + 1e-10)
    return tokens, attr


def build_highlighted_html(text: str, lime_features: list) -> str:
    """Word-level LIME highlighting, HTML-escaped before embedding."""
    word_weights = {word.lower(): weight for word, weight in lime_features}
    max_abs = max((abs(w) for _, w in lime_features), default=1.0)

    parts = [
        '<div style="font-size:16px; line-height:2.2; '
        'font-family:Arial,sans-serif; padding:10px;">'
    ]
    for token in text.split():
        esc = html.escape(token)
        clean_tok = re.sub(r"[^a-z]", "", token.lower())
        if clean_tok in word_weights:
            w = word_weights[clean_tok]
            intensity = min(abs(w) / (max_abs + 1e-10), 1.0)
            alpha = 0.2 + 0.6 * intensity
            if w > 0:  # pushes toward fake
                color = f"rgba(220, 50, 50, {alpha:.3f})"
                border = "2px solid rgba(220,50,50,0.6)"
            else:      # pushes toward real
                color = f"rgba(50, 130, 220, {alpha:.3f})"
                border = "2px solid rgba(50,130,220,0.6)"
            parts.append(
                f'<span style="background:{color}; border-bottom:{border}; '
                f"padding:2px 5px; border-radius:4px; margin:1px; "
                f'display:inline-block" title="LIME weight: {w:+.4f}">{esc}</span>'
            )
        else:
            parts.append(
                f'<span style="padding:2px 3px; display:inline-block">{esc}</span>'
            )
    parts.append("</div>")
    parts.append(
        '<p style="font-size:12px; color:#666; margin-top:8px;">'
        "Red = pushes toward FAKE | "
        "Blue = pushes toward REAL | Hover for LIME weight</p>"
    )
    return "".join(parts)


def run_analysis(review_text: str) -> dict:
    """Full pipeline: preprocess, predict, LIME, IG, reason codes, HTML."""
    start = time.time()
    ms = get_model_service()

    if not review_text or not review_text.strip():
        return {
            "prediction": "N/A",
            "predicted": -1,
            "fake_prob": 0.0,
            "confidence": 0.0,
            "lime_features": [],
            "ig_tokens": [],
            "ig_attrs": [],
            "highlighted_html": "<p>Please enter a review.</p>",
            "reason_codes": [],
            "summary": "No text provided.",
            "elapsed_ms": 0,
        }

    cleaned = preprocess(review_text)

    probs = ms.predict_proba([cleaned])[0]
    pred_label = int(np.argmax(probs))
    fake_prob = float(probs[1])
    confidence = fake_prob if pred_label == 1 else 1 - fake_prob

    exp = ms.lime_explainer.explain_instance(
        cleaned,
        ms.predict_proba,
        num_features=config.LIME_NUM_FEATURES,
        num_samples=config.LIME_NUM_SAMPLES,
        labels=[1],
    )
    lime_features = exp.as_list(label=1)

    ig_tokens, ig_attrs = get_ig_attributions(cleaned, target_class=1)

    rc = generate_reason_codes(
        text=cleaned,
        true_label=-1,  # unknown in real usage
        predicted=pred_label,
        fake_prob=fake_prob,
        lime_exp=exp,
        ig_tokens=ig_tokens,
        ig_attrs=ig_attrs,
    )

    return {
        "prediction": "FAKE" if pred_label == 1 else "REAL",
        "predicted": pred_label,
        "fake_prob": fake_prob,
        "confidence": confidence,
        "lime_features": [
            {"word": word, "weight": float(weight)} for word, weight in lime_features
        ],
        "ig_tokens": ig_tokens,
        "ig_attrs": [float(a) for a in ig_attrs],
        "highlighted_html": build_highlighted_html(cleaned, lime_features),
        "reason_codes": reason_code_items(rc),
        "summary": rc.summary,
        "elapsed_ms": int((time.time() - start) * 1000),
    }


# Re-export for convenience (used by the HITL pool builder too).
__all__ = [
    "run_analysis",
    "get_ig_attributions",
    "build_highlighted_html",
    "get_model_service",
]
