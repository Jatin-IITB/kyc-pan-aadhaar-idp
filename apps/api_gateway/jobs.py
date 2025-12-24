from __future__ import annotations

from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from apps.workers.celery_app import celery_app
from services.ingestion.storage import LocalStorage


def _map_celery_state(state: str) -> str:
    s = (state or "").upper()
    if s in ("PENDING", "RECEIVED", "RETRY"):
        return "QUEUED"
    if s in ("STARTED",):
        return "RUNNING"
    if s in ("SUCCESS",):
        return "SUCCEEDED"
    if s in ("FAILURE", "REVOKED"):
        return "FAILED"
    return "UNKNOWN"


def create_jobs_router(*, storage: LocalStorage) -> APIRouter:
    router = APIRouter()

    @router.post("/jobs")
    async def submit_job(doc_type: str = "auto", file: UploadFile = File(...)):
        job_id = str(uuid4())
        blob = await file.read()

        # store bytes
        stored = storage.put_bytes(job_id=job_id, blob=blob)

        # enqueue task (don’t force task_id)
        async_result = celery_app.send_task(
            "kyc.extract_from_uri",
            args=[job_id, stored.uri, doc_type],
        )

        # persist mapping so GET /jobs/{job_id} can find celery task id
        storage.put_json_atomic(
            job_id=job_id,
            obj={
                "job_id": job_id,
                "celery_task_id": async_result.id,
                "doc_type": doc_type,
                "input_uri": stored.uri,
            },
            name="job_meta.json",
        )

        return JSONResponse(status_code=202, content={"job_id": job_id})

    @router.get("/jobs/{job_id}")
    def job_status(job_id: str):
        meta = storage.get_json_if_exists(job_id=job_id, name="job_meta.json")
        if not meta:
            raise HTTPException(status_code=404, detail="job_not_found")

        task_id = meta.get("celery_task_id")
        if not task_id:
            raise HTTPException(status_code=500, detail="job_corrupt")

        r = AsyncResult(str(task_id), app=celery_app)
        status = _map_celery_state(r.status)

        # Single poll endpoint: include result only if ready
        if status == "SUCCEEDED":
            res = storage.get_json_if_exists(job_id=job_id, name="result.json")
            if res is None:
                # Celery says success but artifact missing => surface as failed
                return {"job_id": job_id, "status": "FAILED", "ok": False, "error": "missing_result_artifact"}
            return {"job_id": job_id, "status": "SUCCEEDED", **res}

        if status == "FAILED":
            err = storage.get_json_if_exists(job_id=job_id, name="error.json")
            if err is None:
                return {"job_id": job_id, "status": "FAILED", "ok": False, "error": "job_failed"}
            return {"job_id": job_id, "status": "FAILED", **err}

        return {"job_id": job_id, "status": status}

    return router


# default router instance
router = create_jobs_router(storage=LocalStorage(root_dir="data/raw/uploads"))
