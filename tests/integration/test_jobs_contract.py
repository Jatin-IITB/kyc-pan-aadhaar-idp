from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api_gateway.jobs import create_jobs_router
from services.ingestion.storage import LocalStorage


class FakeAsyncResult:
    def __init__(self, status: str, result=None):
        self.status = status
        self.result = result


class FakeCelery:
    def __init__(self):
        self.sent = []

    def send_task(self, name: str, args=None, kwargs=None):
        class R:
            id = "fake-task-id-123"
        self.sent.append((name, args, kwargs))
        return R()


def test_jobs_submit_and_poll(monkeypatch, tmp_path):
    # Monkeypatch module globals used by jobs router
    import apps.api_gateway.jobs as jobs_mod

    fake_celery = FakeCelery()
    monkeypatch.setattr(jobs_mod, "celery_app", fake_celery)

    # Make AsyncResult return SUCCESS immediately
    def fake_async_result(task_id: str, app=None):
        return FakeAsyncResult("SUCCESS", result={"ok": True, "result": {"document_type": "pan"}})

    monkeypatch.setattr(jobs_mod, "AsyncResult", fake_async_result)

    storage = LocalStorage(root_dir=str(tmp_path))
    router = create_jobs_router(storage=storage)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    r = client.post("/jobs?doc_type=auto", files={"file": ("x.png", b"abc", "image/png")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # simulate worker wrote result.json
    storage.put_json_atomic(job_id=job_id, obj={"ok": True, "result": {"document_type": "pan"}}, name="result.json")

    r2 = client.get(f"/jobs/{job_id}")
    assert r2.status_code == 200
    j = r2.json()
    assert j["status"] == "SUCCEEDED"
    assert j["ok"] is True
    assert j["result"]["document_type"] == "pan"
