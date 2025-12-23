from __future__ import annotations

import numpy as np
import pytest

import services.pipeline as pipeline_mod


class FakeDetector:
    def __init__(self, fields):
        self._fields = fields

    def detect(self, _img):
        return list(self._fields)


class FakeOCR:
    def __init__(self, results):
        self._results = results

    def extract(self, _img, _fields):
        return list(self._results)


class FakeDocClassifier:
    def __init__(self, route):
        self._route = route

    def predict(self, _img):
        return self._route


class Route:
    def __init__(self, doc_type="unknown", rotation="rot0"):
        self.doc_type = doc_type
        self.rotation = rotation
        self.best_score = 0.0
        self.score_pan = 0.0
        self.score_aadhaar = 0.0
        self.margin = 0.0
        self.confident = False
        self.details = {}


def test_normalize_doc_type():
    assert pipeline_mod.KYCPipeline._normalize_doc_type("aadhar") == "aadhaar"
    assert pipeline_mod.KYCPipeline._normalize_doc_type("aadhaar") == "aadhaar"
    assert pipeline_mod.KYCPipeline._normalize_doc_type("PAN") == "pan"


def test_normalize_key():
    assert pipeline_mod.KYCPipeline._normalize_key("dob") == "date_of_birth"
    assert pipeline_mod.KYCPipeline._normalize_key("aadhar_number") == "aadhaar_number"


def test_collapse_best_per_field_picks_highest_score():
    results = [
        {"field": "dob", "text": "A", "det_conf": 0.9, "ocr_conf": 0.1, "bbox": (0, 0, 1, 1)},
        {"field": "date_of_birth", "text": "B", "det_conf": 0.4, "ocr_conf": 0.9, "bbox": (0, 0, 1, 1)},
    ]
    out = pipeline_mod.KYCPipeline._collapse_best_per_field(results)
    # score(A)=0.5*(0.9+0.1)=0.5 ; score(B)=0.5*(0.4+0.9)=0.65
    assert out["date_of_birth"]["value"] == "B"


def test_extract_from_bgr_unknown_triggers_fallback(monkeypatch):
    # Make schema scoring deterministic and independent of your real schemas/normalizers.
    monkeypatch.setattr(pipeline_mod, "normalize_extraction", lambda ex, dt: ex)
    monkeypatch.setattr(pipeline_mod, "get_required_fields", lambda dt: ["name"])
    monkeypatch.setattr(pipeline_mod, "validate_with_schema", lambda flat, dt: (dt == "pan", "ok"))

    # PAN should win because validate_with_schema returns True for dt=="pan".
    fake_pan_det = FakeDetector(fields=[{"field": "name", "bbox": (0, 0, 10, 10), "conf": 0.9}])
    fake_aad_det = FakeDetector(fields=[{"field": "name", "bbox": (0, 0, 10, 10), "conf": 0.9}])
    fake_ocr = FakeOCR(results=[{"field": "name", "text": "X", "det_conf": 0.9, "ocr_conf": 0.9, "bbox": (0, 0, 10, 10)}])

    pipe = pipeline_mod.KYCPipeline(
        pan_detector=fake_pan_det,
        aadhaar_detector=fake_aad_det,
        ocr=fake_ocr,
        doc_classifier=FakeDocClassifier(Route(doc_type="unknown", rotation="rot0")),
        config=pipeline_mod.PipelineConfig(base_rotations=("rot0",)),
    )

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    out = pipe.extract_from_bgr(img, "auto")
    assert out["document_type"] == "pan"
    assert out["routing_mode"] == "fallback_schema"
