"""Canonical text preprocessing — identical for training and serving.

Port of the notebook's Phase 1 pipeline (cells 10 & 12). The notebook applies
`clean_text()` then `remove_stopwords()` to the training data (cell 12 → cell 16)
and the serving dashboard applies exactly the same two steps (cell 36). Keeping
one function here guarantees train/serve consistency, which is load-bearing.

Deviation from the notebook: stopwords come from scikit-learn's built-in
`ENGLISH_STOP_WORDS` (ships with sklearn, no download) and tokenization uses a
small regex splitter, instead of NLTK's downloaded `stopwords`/`punkt` corpora —
so this module works fully offline. Negation words are kept, matching the
notebook.
"""
from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negation words — they carry sentiment/deception signals.
_KEEP_WORDS = {
    "no", "not", "nor", "never", "neither",
    "none", "nobody", "nothing", "nowhere",
}
_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - _KEEP_WORDS

_TOKEN_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> list[str]:
    """Word tokens (lowercased) — consistent with the cleaned lowercase text."""
    return _TOKEN_RE.findall(text)


def clean_text(text: str) -> str:
    """Remove HTML, normalize unicode, lowercase, strip URLs & special chars."""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    filtered = [word for word in _tokenize(text) if word not in _STOP_WORDS]
    return " ".join(filtered)


def preprocess(text: str) -> str:
    """Canonical full preprocessing used by BOTH training and serving."""
    return remove_stopwords(clean_text(text))
