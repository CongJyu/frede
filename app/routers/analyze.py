"""POST /api/analyze — predict + explain a single review."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import AnalyzeRequest, AnalyzeResponse
from ..services.detector_service import get_detector_service
from ..services.xai_service import run_analysis

router = APIRouter()


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> dict:
    ds = get_detector_service()
    if not ds.ready:
        raise HTTPException(status_code=503, detail=f"Detector not ready: {ds.error}")
    return run_analysis(req.text)
