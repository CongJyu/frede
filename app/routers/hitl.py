"""HITL evaluation endpoints — two-pass human review flow."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import JudgmentRequest
from ..services.hitl_service import HitlUnavailableError, get_hitl_service

router = APIRouter()


def _call(fn):
    try:
        return fn()
    except HitlUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/hitl/session")
def session() -> dict:
    return _call(get_hitl_service().session)


@router.post("/api/hitl/judgment")
def judgment(req: JudgmentRequest) -> dict:
    return _call(lambda: get_hitl_service().submit_judgment(req.judgment, req.confidence, req.feedback))


@router.get("/api/hitl/results")
def results() -> dict:
    return _call(get_hitl_service().results)


@router.post("/api/hitl/reset")
def reset() -> dict:
    get_hitl_service().reset()
    return {"ok": True, "status": "Session reset."}
