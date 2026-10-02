"""Frede FastAPI application entrypoint.

Run:  .venv/bin/python -m uvicorn app.main:app --port 8000 --workers 1
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config
from .routers import analyze, examples, hitl
from .services.detector_service import get_detector_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager load at startup (absorb the model load into startup, not the first
    # request). Run in the executor so the event loop stays free.
    loop = asyncio.get_running_loop()
    ds = get_detector_service()
    try:
        await loop.run_in_executor(None, ds.warmup)
        print(f"[frede] detector ready ({len(ds.features)} features, "
              f"generators={ds.generators})")
    except Exception as exc:  # noqa: BLE001
        # Don't crash the server; /api/health surfaces the error.
        print(f"[frede] detector warmup FAILED: {exc}")

    # The sentiment-proxy model the previous study used is no longer loaded:
    # nothing serves it now that HITL and Examples run on the current detector.
    # `make train` still produces it under models/fake_review_distilbert for
    # reference, but the app does not depend on it.
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Frede — Machine-Generated Review Detector",
        description="GPT-2 surprisal + stylometric features, SHAP and reason codes.",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.include_router(analyze.router)
    app.include_router(examples.router)
    app.include_router(hitl.router)

    @app.get("/api/health")
    def health() -> dict:
        ds = get_detector_service()
        return {
            "status": "ready" if ds.ready else "loading",
            "model": "stage2_detector",
            "features": len(ds.features),
            "generators": ds.generators,
            "error": ds.error,
        }

    @app.middleware("http")
    async def no_cache_spa(request, call_next):
        # The SPA is a small set of static files that change between deploys —
        # force revalidation so browsers never render a stale (pre-fix) version
        # of the HTML/JS/CSS. API responses are left alone.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    # Serve the static SPA last so /api/* routes take precedence.
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
