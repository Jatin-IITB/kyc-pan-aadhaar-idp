from __future__ import annotations

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

celery_app = Celery(
    "kyc_workers",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["apps.workers.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES_S", "86400")),  # 1 day
)
celery_app.conf.broker_connection_retry_on_startup=True


@celery_app.on_after_finalize.connect
def warm_up_models(sender, **kwargs):
    """Pre-load the pipeline graph and VLM model on worker startup."""
    import logging
    _log = logging.getLogger(__name__)
    try:
        from apps.workers.pipeline_loader import get_graph
        graph, deps = get_graph()
        _log.info("Pipeline graph loaded on worker startup")
        if deps.vlm_extractor:
            import urllib.request
            import json
            url = deps.vlm_extractor.config.base_url.rstrip("/") + "/api/generate"
            payload = json.dumps({"model": deps.vlm_extractor.config.model, "prompt": "hello", "stream": False}).encode()
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                urllib.request.urlopen(req, timeout=60)
                _log.info("VLM model %s warmed up", deps.vlm_extractor.config.model)
            except Exception:
                _log.debug("VLM warm-up skipped — Ollama may not be running")
    except Exception as e:
        _log.warning("Worker warm-up failed: %s", e)