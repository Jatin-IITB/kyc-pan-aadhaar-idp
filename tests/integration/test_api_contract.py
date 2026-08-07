from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.app_factory import create_app
from apps.api.jobs import create_jobs_router
from services.ingestion.storage import LocalStorage


def test_health():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_root_redirects_to_docs():
    app = create_app()
    client = TestClient(app, follow_redirects=False)
    r = client.get("/")
    assert r.status_code == 307
    assert "/docs" in r.headers.get("location", "")


def test_submit_job_returns_202(monkeypatch, tmp_path):
    import apps.api.jobs as jobs_mod

    class FakeCelery:
        def send_task(self, name, args=None, kwargs=None):
            class R:
                id = "fake-task-id"
            return R()

    monkeypatch.setattr(jobs_mod, "celery_app", FakeCelery())

    storage = LocalStorage(root_dir=str(tmp_path))
    router = create_jobs_router(storage=storage)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.post("/jobs?doc_type=pan", files={"file": ("test.png", b"fake-image", "image/png")})
    assert r.status_code == 202
    assert "job_id" in r.json()


def test_submit_batch_returns_202(monkeypatch, tmp_path):
    import apps.api.jobs as jobs_mod

    class FakeCelery:
        def send_task(self, name, args=None, kwargs=None):
            class R:
                id = "fake-task-id"
            return R()

    monkeypatch.setattr(jobs_mod, "celery_app", FakeCelery())

    storage = LocalStorage(root_dir=str(tmp_path))
    router = create_jobs_router(storage=storage)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.post(
        "/batches?doc_type=auto",
        files=[
            ("files", ("a.png", b"img1", "image/png")),
            ("files", ("b.png", b"img2", "image/png")),
        ],
    )
    assert r.status_code == 202
    body = r.json()
    assert body["count"] == 2
    assert len(body["job_ids"]) == 2
