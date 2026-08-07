from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from apps.workers.celery_app import celery_app
from services.audit.ledger import AuditLedger
from services.audit.replay import AuditReplayer
from services.ingestion.storage import LocalStorage


def create_cases_router(*, storage: LocalStorage) -> APIRouter:
    router = APIRouter(prefix="/v1/cases", tags=["cases"])

    @router.post("/")
    async def create_case(
        doc_types: str = Query("auto", description="Comma-separated doc types or 'auto'"),
        files: List[UploadFile] = File(...),
    ):
        case_id = str(uuid4())
        doc_type_list = [d.strip() for d in doc_types.split(",")]

        job_ids = []
        for i, f in enumerate(files):
            job_id = str(uuid4())
            blob = await f.read()
            stored = storage.put_bytes(job_id=job_id, blob=blob)

            dt = doc_type_list[i] if i < len(doc_type_list) else "auto"

            async_result = celery_app.send_task(
                "kyc.process_case",
                args=[case_id, job_id, stored.uri, dt],
            )

            storage.put_json_atomic(
                job_id=job_id,
                obj={
                    "job_id": job_id,
                    "case_id": case_id,
                    "celery_task_id": async_result.id,
                    "doc_type": dt,
                    "input_uri": stored.uri,
                    "original_filename": f.filename,
                },
                name="job_meta.json",
            )
            job_ids.append(job_id)

        if len(files) > 1:
            celery_app.send_task(
                "kyc.cross_doc_check",
                args=[case_id, job_ids],
                countdown=15,
            )

        storage.put_json_atomic(
            job_id=f"case_{case_id}",
            obj={
                "case_id": case_id,
                "document_count": len(files),
                "job_ids": job_ids,
            },
            name="case_meta.json",
        )

        return JSONResponse(status_code=202, content={
            "case_id": case_id,
            "document_count": len(files),
            "job_ids": job_ids,
        })

    @router.get("/{case_id}")
    def get_case(case_id: str):
        meta = storage.get_json_if_exists(job_id=f"case_{case_id}", name="case_meta.json")
        if not meta:
            raise HTTPException(status_code=404, detail="case_not_found")

        job_ids = meta.get("job_ids", [])
        documents = []
        for job_id in job_ids:
            job_meta = storage.get_json_if_exists(job_id=job_id, name="job_meta.json")
            result = storage.get_json_if_exists(job_id=job_id, name="result.json")

            celery_task_id = job_meta.get("celery_task_id") if job_meta else None
            r = AsyncResult(str(celery_task_id), app=celery_app) if celery_task_id else None
            status = r.status if r else "UNKNOWN"

            documents.append({
                "job_id": job_id,
                "doc_type": job_meta.get("doc_type", "auto") if job_meta else "unknown",
                "filename": job_meta.get("original_filename") if job_meta else None,
                "status": status,
                "result": result,
            })

        all_done = all(d["status"] in ("SUCCESS", "FAILURE") for d in documents)

        response: Dict[str, Any] = {
            "case_id": case_id,
            "status": "COMPLETED" if all_done else "PROCESSING",
            "document_count": len(documents),
            "documents": documents,
        }

        cross_doc = storage.get_json_if_exists(job_id=f"case_{case_id}", name="cross_doc_result.json")
        if cross_doc:
            response["cross_doc"] = cross_doc

        return response

    @router.get("/{case_id}/documents/{job_id}")
    def get_document(case_id: str, job_id: str):
        meta = storage.get_json_if_exists(job_id=f"case_{case_id}", name="case_meta.json")
        if not meta or job_id not in meta.get("job_ids", []):
            raise HTTPException(status_code=404, detail="document_not_found")

        result = storage.get_json_if_exists(job_id=job_id, name="result.json")
        job_meta = storage.get_json_if_exists(job_id=job_id, name="job_meta.json")

        return {
            "case_id": case_id,
            "job_id": job_id,
            "doc_type": job_meta.get("doc_type") if job_meta else "unknown",
            "result": result,
        }

    @router.get("/{case_id}/audit")
    def get_audit_trail(case_id: str, replay_to: Optional[int] = None):
        audit_data = storage.get_json_if_exists(job_id=f"case_{case_id}", name="audit_events.json")
        if not audit_data:
            raise HTTPException(status_code=404, detail="no_audit_events")

        events = audit_data.get("events", [])

        ledger = AuditLedger()
        verification = ledger.verify_chain(events)

        response: Dict[str, Any] = {
            "case_id": case_id,
            "event_count": len(events),
            "chain_valid": verification["valid"],
            "events": events,
        }

        if replay_to is not None:
            replayer = AuditReplayer()
            response["replayed_state"] = replayer.replay(events, up_to=replay_to)

        return response

    @router.post("/{case_id}/decision")
    async def manual_decision(case_id: str, outcome: str = Query(...), reviewer: str = Query(...)):
        meta = storage.get_json_if_exists(job_id=f"case_{case_id}", name="case_meta.json")
        if not meta:
            raise HTTPException(status_code=404, detail="case_not_found")

        if outcome not in ("APPROVED", "REJECTED", "REQUIRES_INFO"):
            raise HTTPException(status_code=400, detail="invalid_outcome")

        decision = {
            "case_id": case_id,
            "outcome": outcome,
            "reviewer": reviewer,
            "decided_by": "human",
        }

        storage.put_json_atomic(
            job_id=f"case_{case_id}",
            obj=decision,
            name="manual_decision.json",
        )

        return {"case_id": case_id, "outcome": outcome, "reviewer": reviewer}

    return router
