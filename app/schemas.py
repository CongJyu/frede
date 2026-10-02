"""Pydantic request/response schemas for the frede API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    prediction: str          # "MACHINE" | "HUMAN"
    predicted: int           # 1 | 0
    machine_prob: float
    confidence: float
    # `flagged` applies the threshold calibrated at 1% false-positive rate on
    # training folds; `predicted` uses the plain 0.5 boundary. They can differ.
    flagged: bool
    threshold: float
    features: list[dict[str, Any]]   # name, value, human_percentile, shap, direction
    tokens: list[str]
    logprobs: list[float]
    highlighted_html: str
    reason_codes: list[dict[str, Any]]
    summary: str
    elapsed_ms: int
    scope_note: str          # single-generator limitation, surfaced in the UI


class JudgmentRequest(BaseModel):
    # 1 = machine-written, 0 = human-written — the same convention as the labels.
    judgment: str = Field(..., pattern="^(Human|Machine)$")
    confidence: int = Field(..., ge=1, le=5)
    feedback: str = ""


class HealthResponse(BaseModel):
    status: str
    model: str
    error: str | None = None
