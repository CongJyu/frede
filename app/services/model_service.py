"""Process-wide model + tokenizer + LIME explainer singleton.

Loaded once at startup (lifespan warmup) so the first request doesn't pay a
10–30 s model-load cost. Single worker, so a module-level singleton is safe.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .. import config


class ModelService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.lime_explainer = None
        self.device = None
        self.ready = False
        self.loaded_at: float | None = None
        self.error: str | None = None

    def warmup(self) -> None:
        """Load everything. Failures are recorded on the service, not raised,
        so the API stays up (health reports the error)."""
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
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(str(config.MODEL_PATH))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(config.MODEL_PATH)
        )
        self.model.to(self.device)
        self.model.eval()

        self.lime_explainer = LimeTextExplainer(
            class_names=["Real", "Fake"],
            split_expression=r"\W+",
            random_state=config.RANDOM_SEED,
        )
        self.ready = True
        self.loaded_at = time.time()

    def predict_proba(self, texts) -> np.ndarray:
        """(N, 2) array of [P(real), P(fake)] — the format LIME/SHAP expect.

        Accepts a str, list of str, or numpy array of str.
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) if not isinstance(t, str) else t for t in texts]

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), config.BATCH_SIZE):
                batch = texts[i : i + config.BATCH_SIZE]
                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    padding="max_length",
                    max_length=config.MAX_LENGTH,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                # .float() cast avoids the MPS float64 softmax error.
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                outputs.append(probs)
        return np.vstack(outputs).astype(np.float32)


_service = ModelService()


def get_model_service() -> ModelService:
    return _service
