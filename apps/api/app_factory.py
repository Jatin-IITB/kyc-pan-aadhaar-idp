from __future__ import annotations

import os
import time

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from apps.api.auth import require_api_key
from apps.api.cases import create_cases_router
from apps.api.jobs import create_jobs_router
from apps.api.metrics import create_metrics_router, metrics
from apps.api.websocket import create_sse_router
from services.ingestion.storage import LocalStorage


def create_app() -> FastAPI:
    auth_dep = Depends(require_api_key)

    app = FastAPI(title="KYC IDP API Gateway", version="2.0.0")

    allowed_origins = os.environ.get("KYC_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    @app.middleware("http")
    async def track_latency(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        metrics.inc("kyc_api_requests_total")
        metrics.observe("kyc_api_latency_seconds", elapsed)
        return response

    storage = LocalStorage(root_dir="data/raw/uploads")

    app.include_router(create_jobs_router(storage=storage), dependencies=[auth_dep])
    app.include_router(create_cases_router(storage=storage), dependencies=[auth_dep])
    app.include_router(create_metrics_router())
    app.include_router(create_sse_router(), dependencies=[auth_dep])

    @app.get("/")
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app
