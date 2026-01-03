from __future__ import annotations

from typing import List, Dict, Any, Optional
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, UploadFile, Query
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


def get_job_details(job_id: str, storage: LocalStorage, include_result: bool = True) -> Dict[str, Any]:
    """
    Shared logic to fetch status for a single job.
    'include_result': set to False for batch summaries to keep payload light.
    """
    meta = storage.get_json_if_exists(job_id=job_id, name="job_meta.json")
    if not meta:
        return {"job_id": job_id, "status": "UNKNOWN", "error": "not_found"}

    task_id = meta.get("celery_task_id")
    r = AsyncResult(str(task_id), app=celery_app)
    status = _map_celery_state(r.status)

    base_resp = {
        "job_id": job_id,
        "status": status,
        "filename": meta.get("original_filename") or meta.get("filename"),
    }

    if status == "SUCCEEDED":
        # CRITICAL OPTIMIZATION: Only read/return heavy JSON if requested
        if include_result:
            res = storage.get_json_if_exists(job_id=job_id, name="result.json")
            if res is None:
                return {**base_resp, "status": "FAILED", "ok": False, "error": "missing_result_artifact"}
            return {**base_resp, **res}
        else:
            # For batches, just confirm it's done
            return {**base_resp, "ok": True}

    if status == "FAILED":
        err = storage.get_json_if_exists(job_id=job_id, name="error.json")
        if err is None:
            return {**base_resp, "ok": False, "error": "job_failed"}
        return {**base_resp, **err}

    return base_resp


def create_jobs_router(*, storage: LocalStorage) -> APIRouter:
    router = APIRouter()

    # --- Single Job ---
    @router.post("/jobs")
    async def submit_job(doc_type: str = "auto", file: UploadFile = File(...)):
        job_id = str(uuid4())
        blob = await file.read()
        
        stored = storage.put_bytes(job_id=job_id, blob=blob)
        
        async_result = celery_app.send_task(
            "kyc.extract_from_uri",
            args=[job_id, stored.uri, doc_type],
        )

        storage.put_json_atomic(
            job_id=job_id,
            obj={
                "job_id": job_id,
                "celery_task_id": async_result.id,
                "doc_type": doc_type,
                "input_uri": stored.uri,
                "filename": file.filename,
            },
            name="job_meta.json",
        )

        return JSONResponse(status_code=202, content={"job_id": job_id})

    @router.get("/jobs/{job_id}")
    def job_status(job_id: str):
        # Single job poll: we WANT the full result
        return get_job_details(job_id, storage, include_result=True)


    # --- Batch Operations ---
    @router.post("/batches")
    async def submit_batch(
        doc_type: str = Query("auto"),
        files: List[UploadFile] = File(...)
    ):
        batch_id = str(uuid4())
        job_ids = []
        
        batch_meta = {
            "batch_id": batch_id,
            "jobs": [] 
        }

        # In production, you might parallelize these writes using asyncio.gather
        for f in files:
            job_id = str(uuid4())
            blob = await f.read()
            
            stored = storage.put_bytes(job_id=job_id, blob=blob)
            
            async_result = celery_app.send_task(
                "kyc.extract_from_uri",
                args=[job_id, stored.uri, doc_type],
            )
            
            storage.put_json_atomic(
                job_id=job_id,
                obj={
                    "job_id": job_id,
                    "celery_task_id": async_result.id,
                    "doc_type": doc_type,
                    "input_uri": stored.uri,
                    "batch_id": batch_id,
                    "original_filename": f.filename
                },
                name="job_meta.json",
            )
            
            job_ids.append(job_id)
            batch_meta["jobs"].append(job_id)

        # Store batch meta under a "virtual" job ID
        storage.put_json_atomic(
            job_id=f"batch_{batch_id}",
            obj=batch_meta,
            name="batch_meta.json"
        )

        return JSONResponse(
            status_code=202, 
            content={
                "batch_id": batch_id, 
                "count": len(job_ids),
                "job_ids": job_ids
            }
        )

    @router.get("/batches/{batch_id}")
    def batch_status(batch_id: str):
        meta = storage.get_json_if_exists(job_id=f"batch_{batch_id}", name="batch_meta.json")
        if not meta:
            raise HTTPException(status_code=404, detail="batch_not_found")
            
        jobs_list = meta.get("jobs", [])
        
        results = []
        counts = {"QUEUED": 0, "RUNNING": 0, "SUCCEEDED": 0, "FAILED": 0, "UNKNOWN": 0}
        
        for job_id in jobs_list:
            # Lightweight fetch (include_result=False)
            info = get_job_details(job_id, storage, include_result=False)
            
            status = info.get("status", "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
            
            results.append(info)

        total = len(jobs_list)
        # Simple aggregation logic
        if counts["SUCCEEDED"] + counts["FAILED"] == total:
            agg_status = "COMPLETED"
        elif counts["RUNNING"] > 0:
            agg_status = "RUNNING"
        else:
            agg_status = "PENDING"

        return {
            "batch_id": batch_id,
            "status": agg_status,
            "summary": counts,
            "jobs": results 
        }

    return router

# default router instance
router = create_jobs_router(storage=LocalStorage(root_dir="data/raw/uploads"))
