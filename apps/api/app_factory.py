# apps/api/app_factory.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

def create_app() -> FastAPI:
    app = FastAPI(title="KYC IDP API Gateway")

    @app.get("/")
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app
