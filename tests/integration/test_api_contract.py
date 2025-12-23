# tests/integration/test_api_contract.py
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from apps.api_gateway.app_factory import create_app


class FakePipeline:
    def extract_from_bgr(self, img_bgr: np.ndarray, doctype: str):
        return {
            "document_type": "pan" if doctype in ("pan", "auto", "") else doctype,
            "chosen_rotation": "rot0",
            "classifier": None,
            "extraction": {},
            "validation": {"is_valid": True, "message": "Valid"},
            "selection": {"score": 3.0, "coverage": 1.0, "avg_conf": 1.0, "is_valid": True, "message": "Valid"},
            "routing_mode": "detector_then_schema",
        }


def fake_decode(_: bytes) -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_health():
    app = create_app(pipeline=FakePipeline(), decode_fn=fake_decode)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_extract_contract_accepts_doctype_and_doc_type():
    app = create_app(pipeline=FakePipeline(), decode_fn=fake_decode)
    client = TestClient(app)

    r1 = client.post("/extract?doctype=auto", files={"file": ("x.png", b"fake", "image/png")})
    assert r1.status_code == 200

    r2 = client.post("/extract?doc_type=auto", files={"file": ("x.png", b"fake", "image/png")})
    assert r2.status_code == 200

    for r in (r1, r2):
        j = r.json()
        assert "document_type" in j
        assert "chosen_rotation" in j
        assert "extraction" in j
        assert "validation" in j
        assert "is_valid" in j["validation"]
        assert "message" in j["validation"]


def test_extract_batch_contract():
    app = create_app(pipeline=FakePipeline(), decode_fn=fake_decode, max_concurrency=2)
    client = TestClient(app)

    r = client.post(
        "/extract/batch?doctype=auto",
        files=[
            ("files", ("a.png", b"1", "image/png")),
            ("files", ("b.png", b"2", "image/png")),
        ],
    )
    assert r.status_code == 200
    j = r.json()

    assert j["count"] == 2
    assert isinstance(j["results"], list)
    for item in j["results"]:
        assert "filename" in item
        assert "ok" in item
        assert ("result" in item) or ("error" in item)


def test_bad_image_bytes_returns_400():
    def always_fail_decode(_: bytes):
        raise ValueError("nope")

    app = create_app(pipeline=FakePipeline(), decode_fn=always_fail_decode)
    client = TestClient(app)

    r = client.post(
        "/extract?doctype=auto",
        files={"file": ("x.bin", b"not-an-image", "application/octet-stream")},
    )
    assert r.status_code == 400
