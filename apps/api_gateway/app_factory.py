# apps/api_gateway/app_factory.py
from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any, Callable, Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import RedirectResponse
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool


def decode_image_with_exif(contents: bytes) -> np.ndarray:
    img_pil = Image.open(BytesIO(contents))
    img_pil = ImageOps.exif_transpose(img_pil)
    img_rgb = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def create_app(
    *,
    pipeline: Any,
    decode_fn: Callable[[bytes], np.ndarray] = decode_image_with_exif,
    max_concurrency: int = 4,
) -> FastAPI:
    app = FastAPI(title="KYC IDP API Gateway")

    def extract_single_from_contents(contents: bytes, doctype: str) -> Dict[str, Any]:
        try:
            img = decode_fn(contents)
        except Exception:
            img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        try:
            return pipeline.extract_from_bgr(img, doctype)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/")
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.post("/extract")
    async def extract_kyc_data(
        doctype: str = Query("auto", alias="doctype"),
        doc_type: str = Query("", alias="doc_type"),
        file: UploadFile = File(...),
    ):
        # Prefer doctype; fall back to doc_type if provided.
        dt = doctype if doctype else (doc_type if doc_type else "auto")
        contents = await file.read()
        return extract_single_from_contents(contents, dt)

    @app.post("/extract/batch")
    async def extract_kyc_data_batch(
        doctype: str = Query("auto", alias="doctype"),
        doc_type: str = Query("", alias="doc_type"),
        files: List[UploadFile] = File(...),
    ):
        # /extract/batch is multipart “files”, consistent with your gateway contract. [file:309]
        dt = doctype if doctype else (doc_type if doc_type else "auto")
        sem = asyncio.Semaphore(max_concurrency)

        async def one(f: UploadFile):
            async with sem:
                try:
                    contents = await f.read()
                    out = await run_in_threadpool(extract_single_from_contents, contents, dt)
                    return {"filename": f.filename, "ok": True, "result": out}
                except HTTPException as e:
                    return {"filename": f.filename, "ok": False, "error": e.detail}
                except Exception as e:
                    return {"filename": f.filename, "ok": False, "error": str(e)}

        results = await asyncio.gather(*(one(f) for f in files))
        return {"count": len(results), "results": results}

    return app
