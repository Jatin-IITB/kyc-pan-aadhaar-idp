from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from apps.workers.celery_app import celery_app
from celery.result import AsyncResult


def create_sse_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/cases/{case_id}/progress")
    async def case_progress(case_id: str):
        async def event_stream():
            from services.ingestion.storage import LocalStorage
            storage = LocalStorage(root_dir="data/raw/uploads")

            meta = storage.get_json_if_exists(job_id=f"case_{case_id}", name="case_meta.json")
            if not meta:
                yield _sse_event("error", {"message": "case_not_found"})
                return

            job_ids = meta.get("job_ids", [])
            completed = set()

            yield _sse_event("started", {
                "case_id": case_id,
                "document_count": len(job_ids),
            })

            for _ in range(120):
                for job_id in job_ids:
                    if job_id in completed:
                        continue

                    job_meta = storage.get_json_if_exists(job_id=job_id, name="job_meta.json")
                    if not job_meta:
                        continue

                    task_id = job_meta.get("celery_task_id")
                    if not task_id:
                        continue

                    r = AsyncResult(str(task_id), app=celery_app)
                    state = (r.status or "").upper()

                    if state == "SUCCESS":
                        completed.add(job_id)
                        yield _sse_event("document_complete", {
                            "job_id": job_id,
                            "status": "SUCCESS",
                            "progress": f"{len(completed)}/{len(job_ids)}",
                        })
                    elif state in ("FAILURE", "REVOKED"):
                        completed.add(job_id)
                        yield _sse_event("document_complete", {
                            "job_id": job_id,
                            "status": "FAILED",
                            "progress": f"{len(completed)}/{len(job_ids)}",
                        })

                if len(completed) == len(job_ids):
                    yield _sse_event("case_complete", {
                        "case_id": case_id,
                        "total": len(job_ids),
                        "succeeded": sum(1 for jid in completed if _check_success(jid)),
                    })
                    return

                await asyncio.sleep(1)

            yield _sse_event("timeout", {"case_id": case_id})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _check_success(job_id: str) -> bool:
    from services.ingestion.storage import LocalStorage
    storage = LocalStorage(root_dir="data/raw/uploads")
    job_meta = storage.get_json_if_exists(job_id=job_id, name="job_meta.json")
    if not job_meta:
        return False
    task_id = job_meta.get("celery_task_id")
    if not task_id:
        return False
    r = AsyncResult(str(task_id), app=celery_app)
    return (r.status or "").upper() == "SUCCESS"
