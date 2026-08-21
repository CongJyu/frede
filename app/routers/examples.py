"""GET /api/examples — precomputed SHAP/IG/LIME examples for the Examples page."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from .. import config

router = APIRouter()


@router.get("/api/examples")
def get_examples() -> dict:
    if not config.EXAMPLES_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Examples missing at {config.EXAMPLES_PATH}. Run `make train`.",
        )
    with open(config.EXAMPLES_PATH, encoding="utf-8") as f:
        return json.load(f)
