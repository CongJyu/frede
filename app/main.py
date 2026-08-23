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
from .services.model_service import get_model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager model load at startup (absorb the ~10–30 s load into startup, not
    # the first request). Run in the executor so the event loop stays free.
    ms = get_model_service()
    try:
        await asyncio.get_running_loop().run_in_executor(None, ms.warmup)
        print(f"[frede] model ready (device={ms.device})")
    except Exception as exc:  # noqa: BLE001
        # Don't crash the server; /api/health surfaces the error.
        print(f"[frede] model warmup FAILED: {exc}")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Frede — Fake Restaurant Review Detector",
        description="Fine-tuned transformer + LIME / Integrated Gradients / Reason Codes.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(analyze.router)
    app.include_router(examples.router)
    app.include_router(hitl.router)

    @app.get("/api/health")
    def health() -> dict:
        ms = get_model_service()
        return {
            "status": "ready" if ms.ready else "loading",
            "model": config.MODEL_NAME,
            "error": ms.error,
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
