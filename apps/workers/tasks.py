from __future__ import annotations

import cv2
import numpy as np

from apps.workers.celery_app import celery_app
from apps.workers.pipeline_loader import get_pipeline
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

        try:
            img = decode_image_with_exif(blob)
        except Exception:
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise BadInputError("could not decode image")

        pipe = get_pipeline()

        # If you later see random Paddle/torch hiccups, wrap them:
        try:
            result = pipe.extract_from_bgr(img, doc_type)
        except Exception as e:
            raise TransientWorkerError(str(e)) from e

        payload = {"ok": True, "result": result}
        storage.put_json_atomic(job_id=job_id, obj=payload, name="result.json")
        return payload

    except BadInputError as e:
        payload = {"ok": False, "error": "bad_input", "detail": str(e)[:300]}
        storage.put_json_atomic(job_id=job_id, obj=payload, name="error.json")
        return payload

    except Exception as e:
        payload = {"ok": False, "error": "job_failed", "detail": str(e)[:300]}
        storage.put_json_atomic(job_id=job_id, obj=payload, name="error.json")
        return payload
