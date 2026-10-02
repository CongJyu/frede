"""Serving singleton for the Stage 2 machine-generated-review detector.

Unlike the shipped `model_service` (a fine-tuned transformer over preprocessed
text), this detector scores **raw** text through two feature families — GPT-2
surprisal and stylometric surface statistics — and a gradient-boosted
classifier. `app.preprocess.preprocess` is deliberately not applied: Stage 0
measured that it deletes ~0.19 AUC of the available signal.

It also carries the human/machine reference profile produced at training time,
because a feature value on its own says nothing. "Mean surprisal −3.1" only
becomes an explanation once you can say it sits at the 4th percentile of real
reviews.
"""
from __future__ import annotations

import threading
import time

import joblib
import numpy as np
import pandas as pd

from .. import config
from detect.features import SurprisalExtractor, build_feature_row, stylometric_features


class DetectorService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.extractor: SurprisalExtractor | None = None
        self.model = None
        self.features: list[str] = []
        self.reference: dict = {}
        self.shap_explainer = None
        self.threshold: float = 0.5
        self.generators: list[str] = []
        self.ready = False
        self.error: str | None = None

    def warmup(self) -> None:
        if self.ready:
            return
        with self._lock:
            if self.ready:
                return
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                raise

    def _load(self) -> None:
        import shap

        bundle = joblib.load(config.DETECTOR_PATH)
        self.model = bundle["model"]
        self.features = list(bundle["features"])
        self.threshold = float(bundle["threshold_at_fpr_1pct"])
        self.generators = list(bundle.get("generators", []))

        import json

        with open(config.REFERENCE_PATH, encoding="utf-8") as f:
            self.reference = json.load(f)["profile"]

        self.extractor = SurprisalExtractor(
            model_name=config.SURPRISAL_LM,
            device="mps" if _mps() else "cpu",
            max_length=config.SURPRISAL_MAX_LENGTH,
            cache_dir=str(config.DATA_DIR / "hf_cache"),
        )
        # TreeExplainer on a tree ensemble is exact and returns in ~30 ms, which
        # is why this path needs no LIME-style sampling approximation at all.
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.ready = True

    def score(self, text: str) -> dict:
        """Feature extraction, probability and per-feature SHAP for one review."""
        start = time.time()
        surp, tokens, logprobs = self.extractor.analyze(text)
        row = build_feature_row(surp, text)

        X = pd.DataFrame([{c: row.get(c, np.nan) for c in self.features}])
        prob = float(self.model.predict_proba(X)[:, 1][0])
        shap_values = np.asarray(self.shap_explainer.shap_values(X)).ravel()

        return {
            "prob": prob,
            "features": {c: float(row.get(c, np.nan)) for c in self.features},
            "shap": {c: float(v) for c, v in zip(self.features, shap_values)},
            "tokens": tokens,
            "logprobs": logprobs,
            "stylometric": stylometric_features(text),
            "elapsed_ms": int((time.time() - start) * 1000),
        }

    def human_percentile(self, feature: str, value: float) -> float | None:
        """Where `value` falls in the distribution of real human reviews.

        Returned as a percentile so a reason code can say "more predictable than
        99% of real reviews" rather than quoting a raw log-probability.
        """
        ref = self.reference.get(feature, {}).get("human_pct")
        if not ref or not np.isfinite(value):
            return None
        return float(np.interp(value, ref, [1, 5, 25, 50, 75, 95, 99]))


_service = DetectorService()


def get_detector_service() -> DetectorService:
    return _service


def _mps() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        return False
