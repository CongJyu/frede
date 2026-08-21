"""Precompute Examples-page data: TP/TN/FP/FN samples with IG + LIME + SHAP.

SHAP is computed offline here (partition explainer) because it is far too slow
to run per-request. If shap/numba are unavailable, the page degrades to IG+LIME.
"""
from __future__ import annotations

import json

import numpy as np
from lime.lime_text import LimeTextExplainer

from app.reason_codes import generate_reason_codes, reason_code_items
from app.services.xai_service import build_highlighted_html, get_ig_attributions
from . import config as tc
from .hitl_pool import make_predictor


def _make_shap_explainer(predictor):
    try:
        import shap
        from shap.maskers import Text as ShapTextMasker

        return shap.Explainer(
            predictor,
            ShapTextMasker(tokenizer=r"\s+"),
            output_names=["Real", "Fake"],
            algorithm="partition",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[SHAP unavailable, skipping: {exc}]")
        return None


def build_examples(
        model, tokenizer, device, X_test, y_test,
        all_preds, all_probs, all_labels, with_shap=False,
) -> None:
    all_preds_arr = np.asarray(all_preds)
    all_labels_arr = np.asarray(all_labels)
    tp = np.where((all_labels_arr == 1) & (all_preds_arr == 1))[0]
    tn = np.where((all_labels_arr == 0) & (all_preds_arr == 0))[0]
    fp = np.where((all_labels_arr == 0) & (all_preds_arr == 1))[0]
    fn = np.where((all_labels_arr == 1) & (all_preds_arr == 0))[0]

    picks = [
        ("True Positive — fake correctly detected", tp[0] if len(tp) else 0),
        ("True Negative — real correctly detected", tn[0] if len(tn) else 1),
        ("False Positive — real misclassified as fake", fp[0] if len(fp) else 2),
        ("False Negative — fake missed by the model", fn[0] if len(fn) else 3),
    ]

    predictor = make_predictor(model, tokenizer, device, tc.MAX_LENGTH, tc.BATCH_SIZE)
    explainer = LimeTextExplainer(
        class_names=["Real", "Fake"], split_expression=r"\W+", random_state=tc.RANDOM_SEED
    )
    shap_explainer = _make_shap_explainer(predictor) if with_shap else None

    examples = []
    for title, idx in picks:
        text = X_test[int(idx)]
        true_label = int(y_test[int(idx)])
        pred_label = int(all_preds[int(idx)])
        fake_prob = float(all_probs[int(idx)])
        truncated = " ".join(text.split()[:100])

        exp = explainer.explain_instance(
            text, predictor,
            num_features=tc.LIME_NUM_FEATURES,
            num_samples=tc.LIME_NUM_SAMPLES,
            labels=[1],
        )
        lime_features = exp.as_list(label=1)
        ig_tokens, ig_attrs = get_ig_attributions(
            text, target_class=1, model=model, tokenizer=tokenizer, device=device
        )
        rc = generate_reason_codes(
            text=text, true_label=true_label, predicted=pred_label,
            fake_prob=fake_prob, lime_exp=exp,
            ig_tokens=ig_tokens, ig_attrs=ig_attrs,
        )

        shap_data = None
        if shap_explainer is not None:
            try:
                sv = shap_explainer([truncated])
                vals = sv[0].values
                vals = vals[:, 1] if vals.ndim == 2 else vals
                shap_data = {
                    "tokens": list(sv[0].data),
                    "values": [float(v) for v in vals],
                }
            except Exception as exc:  # noqa: BLE001
                print(f"  [SHAP for '{title[:40]}' failed: {exc}]")

        examples.append(
            {
                "title": title,
                "text": text,
                "true_label": true_label,
                "model_pred": pred_label,
                "fake_prob": fake_prob,
                "reason_codes": reason_code_items(rc),
                "summary": rc.summary,
                "highlighted_html": build_highlighted_html(text, lime_features),
                "lime_features": [
                    {"word": w, "weight": float(weight)} for w, weight in lime_features
                ],
                "ig_tokens": ig_tokens,
                "ig_attrs": [float(a) for a in ig_attrs],
                "shap": shap_data,
            }
        )
        print(
            f"  Example '{title}': pred={'Fake' if pred_label else 'Real'} "
            f"(true={'Fake' if true_label else 'Real'})"
        )

    tc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = tc.DATA_DIR / "examples.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"examples": examples}, f, ensure_ascii=False, indent=2)
    print(f"Examples saved -> {path}")
