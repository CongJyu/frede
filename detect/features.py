"""Zero-shot machine-generated-text signals for review text.

Every feature here is computed on **RAW** review text. Do NOT feed these the
output of `app.preprocess.preprocess`: that pipeline lowercases, strips all
punctuation/digits/casing and removes stopwords — i.e. it deletes precisely the
signals that separate human from machine writing. Stage 0 measures that loss
explicitly; see `detect/stage0.py`.

Two independent families:

1. `SurprisalExtractor` — statistics of per-token log-probability under a
   reference causal LM. Includes the sampling-free *conditional probability
   curvature* of Fast-DetectGPT (Bao et al., 2024): machine text sits near a
   local maximum of the reference model's likelihood, so the observed
   log-likelihood is high relative to what sampling from the model would give.

2. `stylometric_features` — surface statistics: sentence-length burstiness,
   contraction rate, punctuation/casing regularity, and so on.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class SurprisalFeatures:
    """Per-document statistics of token surprisal under a reference LM."""

    lp_mean: float        # mean log p(x_i | x_<i): how predictable the text is
    lp_std: float         # spread of surprisal ("burstiness" of information)
    lp_min: float         # least predictable token
    lp_max: float         # most predictable token
    curvature: float      # Fast-DetectGPT conditional probability curvature
    frac_top1: float      # share of tokens the model's greedy pick got right
    mean_entropy: float   # mean predictive entropy along the text
    n_tokens: int

    @property
    def ppl(self) -> float:
        """Perplexity of the text under the reference model."""
        return float(math.exp(-self.lp_mean))


class SurprisalExtractor:
    """Batched surprisal features under a causal LM, memory-bounded in T.

    The vocabulary moment computation materialises (B, T, V) tensors, which for
    GPT-2 (V=50257) is ~50 MB per 256 positions per sequence. We therefore walk
    the position axis in `pos_chunk` blocks rather than holding the whole
    (B, T, V) at once.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str | None = None,
        max_length: int = 256,
        cache_dir: str | None = None,
        batch_size: int = 4,
        pos_chunk: int = 64,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pos_chunk = pos_chunk

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def extract(self, texts: list[str], batch_size: int | None = None):
        """Yield `SurprisalFeatures` for each text, in input order."""
        batch_size = batch_size or self.batch_size
        out: list[SurprisalFeatures] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._extract_batch(texts[i : i + batch_size]))
        return out

    @torch.no_grad()
    def analyze(self, text: str):
        """One forward pass returning both the summary and per-token detail.

        Serving needs the token-level view for the explanation, and recomputing
        the forward pass to get it would double the per-request cost. Returns
        (SurprisalFeatures, tokens, logprobs) where `logprobs[i]` is
        log p(token_i | token_<i).
        """
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return self._empty(), [], []

        logits = self.model(input_ids=input_ids).logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        lg = shift_logits - torch.logsumexp(shift_logits, dim=-1, keepdim=True)
        p = lg.exp()
        mu = (p * lg).sum(-1)
        var = ((p * lg * lg).sum(-1) - mu.pow(2)).clamp_min(0)
        lp = lg.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        lp_flat, mu_flat, var_flat = lp[0], mu[0], var[0]
        sum_lp, sum_var = float(lp_flat.sum()), float(var_flat.sum())
        summary = SurprisalFeatures(
            lp_mean=float(lp_flat.mean()),
            lp_std=float(lp_flat.std(unbiased=False)),
            lp_min=float(lp_flat.min()),
            lp_max=float(lp_flat.max()),
            curvature=(sum_lp - float(mu_flat.sum())) / math.sqrt(sum_var + 1e-8),
            frac_top1=float((lg[0].argmax(-1) == shift_labels[0]).float().mean()),
            mean_entropy=-float(mu_flat.mean()),
            n_tokens=int(lp_flat.numel()),
        )

        # Token 0 has no preceding context, so its probability is undefined here;
        # drop it and align the arrays with the tokens actually scored.
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, 1:].tolist())
        return summary, tokens, [float(x) for x in lp_flat]

    @torch.no_grad()
    def _extract_batch(self, texts: list[str]) -> list[SurprisalFeatures]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        # A 1-token sequence has no predictable position at all.
        if input_ids.shape[1] < 2:
            return [self._empty() for _ in texts]

        logits = self.model(input_ids=input_ids, attention_mask=attn).logits

        # Token at position t (t >= 1) is predicted by logits at position t - 1.
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        valid = attn[:, 1:].bool()

        n_pos = shift_labels.shape[1]
        lp = torch.zeros_like(shift_labels, dtype=torch.float32)
        mu = torch.zeros_like(lp)
        var = torch.zeros_like(lp)
        top1 = torch.zeros_like(valid)

        for t0 in range(0, n_pos, self.pos_chunk):
            t1 = min(t0 + self.pos_chunk, n_pos)
            lg = shift_logits[:, t0:t1, :].float()
            lg = lg - torch.logsumexp(lg, dim=-1, keepdim=True)  # log_softmax
            p = lg.exp()
            # mu = E[log p] over the model's own predictive distribution
            mu[:, t0:t1] = (p * lg).sum(-1)
            # var = E[(log p)^2] - mu^2  (clamped: float error can go negative)
            var[:, t0:t1] = ((p * lg * lg).sum(-1) - mu[:, t0:t1].pow(2)).clamp_min(0)
            lp[:, t0:t1] = lg.gather(-1, shift_labels[:, t0:t1].unsqueeze(-1)).squeeze(-1)
            top1[:, t0:t1] = lg.argmax(-1) == shift_labels[:, t0:t1]
            del lg, p

        results: list[SurprisalFeatures] = []
        for b in range(len(texts)):
            m = valid[b]
            n = int(m.sum().item())
            if n == 0:
                results.append(self._empty())
                continue
            lp_b, mu_b, var_b = lp[b][m], mu[b][m], var[b][m]
            sum_lp, sum_var = float(lp_b.sum()), float(var_b.sum())
            results.append(
                SurprisalFeatures(
                    lp_mean=float(lp_b.mean()),
                    lp_std=float(lp_b.std(unbiased=False)),
                    lp_min=float(lp_b.min()),
                    lp_max=float(lp_b.max()),
                    # Fast-DetectGPT curvature: how far above the model's own
                    # sampling expectation this text's log-likelihood sits.
                    curvature=(sum_lp - float(mu_b.sum())) / math.sqrt(sum_var + 1e-8),
                    frac_top1=float(top1[b][m].float().mean()),
                    # mu is sum_v p log p = -H(p), so -mean(mu) is mean entropy.
                    mean_entropy=-float(mu_b.mean()),
                    n_tokens=n,
                )
            )
        return results

    @staticmethod
    def _empty() -> SurprisalFeatures:
        nan = float("nan")
        return SurprisalFeatures(nan, nan, nan, nan, nan, nan, nan, 0)


# --------------------------------------------------------------------------
# Stylometric features
# --------------------------------------------------------------------------

_SENT_RE = re.compile(r"[^.!?\n]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_CONTRACTION_RE = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.I)
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")

STYLOMETRIC_NAMES = [
    "n_words",
    "sent_len_mean",
    "sent_len_std",
    "sent_len_cv",
    "ttr",
    "avg_word_len",
    "contraction_rate",
    "comma_per_sent",
    "excl_per_sent",
    "quest_per_sent",
    "upper_ratio",
    "allcaps_rate",
    "digit_per_sent",
    "newline_per_sent",
    "has_em_dash",
]


def stylometric_features(text: str) -> dict[str, float]:
    """Surface-style statistics. Cheap, deterministic, no model needed."""
    sentences = [s for s in _SENT_RE.findall(text) if s.strip()]
    words = _WORD_RE.findall(text)
    n_words = len(words)
    n_sents = len(sentences)
    sent_lens = [len(_WORD_RE.findall(s)) for s in sentences]
    sent_lens = [n for n in sent_lens if n > 0]

    sl_mean = float(np.mean(sent_lens)) if sent_lens else 0.0
    sl_std = float(np.std(sent_lens)) if sent_lens else 0.0
    # Coefficient of variation = "burstiness": humans mix 3-word and 60-word
    # sentences; machine text is far more uniform.
    sl_cv = sl_std / sl_mean if sl_mean > 0 else 0.0

    alpha = [c for c in text if c.isalpha()]
    return {
        "n_words": float(n_words),
        "sent_len_mean": sl_mean,
        "sent_len_std": sl_std,
        "sent_len_cv": sl_cv,
        "ttr": len({w.lower() for w in words}) / n_words if n_words else 0.0,
        "avg_word_len": float(np.mean([len(w) for w in words])) if n_words else 0.0,
        "contraction_rate": (
            len(_CONTRACTION_RE.findall(text)) / n_words if n_words else 0.0
        ),
        "comma_per_sent": text.count(",") / n_sents if n_sents else 0.0,
        "excl_per_sent": text.count("!") / n_sents if n_sents else 0.0,
        "quest_per_sent": text.count("?") / n_sents if n_sents else 0.0,
        "upper_ratio": (
            sum(1 for c in alpha if c.isupper()) / len(alpha) if alpha else 0.0
        ),
        "allcaps_rate": (
            len(_ALL_CAPS_RE.findall(text)) / n_words if n_words else 0.0
        ),
        "digit_per_sent": sum(c.isdigit() for c in text) / n_sents if n_sents else 0.0,
        "newline_per_sent": text.count("\n") / n_sents if n_sents else 0.0,
        "has_em_dash": float("—" in text or " - " in text),
    }


def all_feature_names() -> list[str]:
    """Stable column order for the combined feature table."""
    return [f"surp_{k}" for k in asdict(SurprisalExtractor._empty()).keys()] + [
        f"style_{k}" for k in STYLOMETRIC_NAMES
    ]


def build_feature_row(surp: SurprisalFeatures, text: str) -> dict[str, float]:
    """Flatten one document into a single flat feature dict."""
    row = {f"surp_{k}": v for k, v in asdict(surp).items()}
    row["surp_ppl"] = surp.ppl
    row.update({f"style_{k}": v for k, v in stylometric_features(text).items()})
    return row
