"""Precompute the balanced HITL sample pool with explanations.

Port of notebook Phase 4 Part 2. Explanations are computed offline at train time
so the HITL UI serves instantly with zero live LIME.
"""
from __future__ import annotations

import json

import numpy as np
from lime.lime_text import LimeTextExplainer

from app.reason_codes import generate_reason_codes, reason_code_items
from app.services.xai_service import build_highlighted_html
from . import config as tc


def make_predictor(model, tokenizer, device, max_length, batch_size):
    """Batch predict-proba closure bound to the freshly-trained model."""
    import torch

    def predict_proba(texts) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                out.append(p)
        return np.vstack(out).astype(np.float32)

    return predict_proba


def build_hitl_pool(model, tokenizer, device, X_test, y_test, all_preds, all_probs) -> None:
    rng = np.random.RandomState(tc.HITL_SEED)
    fake_idx = [i for i, label in enumerate(y_test) if label == 1]
    real_idx = [i for i, label in enumerate(y_test) if label == 0]
    n_each = tc.HITL_N_SAMPLES // 2
    sampled_fake = rng.choice(fake_idx, size=min(n_each, len(fake_idx)), replace=False)
    sampled_real = rng.choice(real_idx, size=min(n_each, len(real_idx)), replace=False)
    hitl_idx = np.concatenate([sampled_fake, sampled_real])
    rng.shuffle(hitl_idx)

    predictor = make_predictor(model, tokenizer, device, tc.MAX_LENGTH, tc.BATCH_SIZE)
    explainer = LimeTextExplainer(
        class_names=["Real", "Fake"], split_expression=r"\W+", random_state=tc.RANDOM_SEED
    )

    samples = []
    for idx in hitl_idx:
        text = X_test[int(idx)]
        true_label = int(y_test[int(idx)])
        pred_label = int(all_preds[int(idx)])
        fake_prob = float(all_probs[int(idx)])

        exp = explainer.explain_instance(
            text,
            predictor,
            num_features=tc.LIME_NUM_FEATURES,
            num_samples=tc.LIME_NUM_SAMPLES,
            labels=[1],
        )
        lime_features = exp.as_list(label=1)
        rc = generate_reason_codes(
            text=text, true_label=true_label, predicted=pred_label,
            fake_prob=fake_prob, lime_exp=exp,
        )

        samples.append(
            {
                "idx": int(idx),
                "text": text,
                "true_label": true_label,
                "model_pred": pred_label,
                "fake_prob": fake_prob,
                "highlighted_html": build_highlighted_html(text, lime_features),
                "reason_codes": reason_code_items(rc),
                "summary": rc.summary,
            }
        )
        print(
            f"  HITL sample {len(samples)}: "
            f"pred={'Fake' if pred_label else 'Real'} P(fake)={fake_prob:.3f} "
            f"RC={[r['code'] for r in reason_code_items(rc)]}"
        )

    tc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = tc.DATA_DIR / "hitl_pool.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"HITL pool saved: {len(samples)} samples -> {path}")
