"""Pydantic request/response schemas for the frede API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    prediction: str          # "FAKE" | "REAL"
    predicted: int           # 1 | 0
    fake_prob: float
    confidence: float
    lime_features: list[dict[str, Any]]
    ig_tokens: list[str]
    ig_attrs: list[float]
    highlighted_html: str
    reason_codes: list[dict[str, Any]]
    summary: str
    elapsed_ms: int


class JudgmentRequest(BaseModel):
    judgment: str = Field(..., pattern="^(Real|Fake)$")
    confidence: int = Field(..., ge=1, le=5)
    feedback: str = ""


class HealthResponse(BaseModel):
    status: str
    model: str
    error: str | None = None
