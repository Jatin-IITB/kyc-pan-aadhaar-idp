from __future__ import annotations

import cv2
import numpy as np

from apps.workers.celery_app import celery_app
from apps.workers.pipeline_loader import check_hot_reload, get_graph, get_pipeline
from services.graph.workflow import invoke_graph
from services.ingestion.storage import LocalStorage

storage = LocalStorage(root_dir="data/raw/uploads")


class BadInputError(ValueError):
    """Non-retriable."""


class TransientWorkerError(RuntimeError):
    """Retriable."""


def decode_image_with_exif(contents: bytes) -> np.ndarray:
    from io import BytesIO
    from PIL import Image, ImageOps

    img_pil = Image.open(BytesIO(contents))
    img_pil = ImageOps.exif_transpose(img_pil)
    img_rgb = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _decode_image(blob: bytes) -> np.ndarray:
    try:
        return decode_image_with_exif(blob)
    except Exception:
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise BadInputError("could not decode image")
        return img


@celery_app.task(
    name="kyc.extract_from_uri",
    bind=True,
    autoretry_for=(TransientWorkerError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def extract_from_uri(self, job_id: str, input_uri: str, doc_type: str = "auto") -> dict:
    try:
        blob = storage.get_bytes(uri=input_uri)
        img = _decode_image(blob)

        pipe = get_pipeline()
        try:
            result = pipe.extract_from_bgr(img, doc_type)
        except Exception as e:
            raise TransientWorkerError(str(e)) from e

        payload = {"ok": True, "result": result}
        storage.put_json_atomic(job_id=job_id, obj=payload, name="result.json")
        return payload

    except BadInputError as e:
        err = {"ok": False, "error": "bad_input", "detail": str(e)[:300]}
        storage.put_json_atomic(job_id=job_id, obj=err, name="error.json")
        return err

    except Exception as e:
        err = {"ok": False, "error": "job_failed", "detail": str(e)[:300]}
        storage.put_json_atomic(job_id=job_id, obj=err, name="error.json")
        return err


@celery_app.task(
    name="kyc.process_case",
    bind=True,
    autoretry_for=(TransientWorkerError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def process_case(self, case_id: str, job_id: str, input_uri: str, doc_type: str = "auto") -> dict:
    check_hot_reload()
    try:
        blob = storage.get_bytes(uri=input_uri)
        img = _decode_image(blob)

        compiled_graph, deps = get_graph()

        initial_state = {
            "case_id": case_id,
            "image_bgr": img,
            "requested_doc_type": doc_type,
        }

        try:
            final_state = invoke_graph(compiled_graph, deps, initial_state)
        except Exception as e:
            raise TransientWorkerError(str(e)) from e

        result = final_state.get("final_result", {})
        audit_events = final_state.get("audit_events", [])
        payload = {"ok": True, "result": result}
        storage.put_json_atomic(job_id=job_id, obj=payload, name="result.json")

        if audit_events:
            storage.put_json_atomic(
                job_id=f"case_{case_id}",
                obj={"events": audit_events},
                name="audit_events.json",
            )

        return payload

    except BadInputError as e:
        err = {"ok": False, "error": "bad_input", "detail": str(e)[:300]}
        storage.put_json_atomic(job_id=job_id, obj=err, name="error.json")
        return err

    except Exception as e:
        err = {"ok": False, "error": "job_failed", "detail": str(e)[:300]}
        storage.put_json_atomic(job_id=job_id, obj=err, name="error.json")
        return err


@celery_app.task(
    name="kyc.cross_doc_check",
    bind=True,
    max_retries=10,
    default_retry_delay=5,
)
def cross_doc_check(self, case_id: str, job_ids: list) -> dict:
    from services.cross_doc.contradiction import ContradictionDetector

    documents = []
    for job_id in job_ids:
        result_data = storage.get_json_if_exists(job_id=job_id, name="result.json")
        if not result_data:
            raise self.retry(countdown=5)
        result = result_data.get("result", {})
        documents.append({
            "job_id": job_id,
            "doc_type": result.get("document_type", "unknown"),
            "extraction": result.get("extraction", {}),
        })

    detector = ContradictionDetector()
    fields_docs = []
    for doc in documents:
        ext = doc.get("extraction", {})
        fields = {}
        for k, v in ext.items():
            if isinstance(v, dict) and "value" in v:
                fields[k] = v["value"]
            elif isinstance(v, str):
                fields[k] = v
        fields_docs.append({"doc_type": doc["doc_type"], "fields": fields})

    cross_doc_result = detector.detect(fields_docs)

    storage.put_json_atomic(
        job_id=f"case_{case_id}",
        obj=cross_doc_result,
        name="cross_doc_result.json",
    )
    return cross_doc_result
