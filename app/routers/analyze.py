"""POST /api/analyze — predict + explain a single review."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import AnalyzeRequest, AnalyzeResponse
from ..services.model_service import get_model_service
from ..services.xai_service import run_analysis

router = APIRouter()


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> dict:
    ms = get_model_service()
    if not ms.ready:
        raise HTTPException(status_code=503, detail=f"Model not ready: {ms.error}")
    return run_analysis(req.text)
