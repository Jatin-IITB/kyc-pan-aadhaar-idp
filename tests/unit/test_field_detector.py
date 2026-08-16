"""Tests for FieldDetector field name mapping and _NullDetector."""
from __future__ import annotations

import numpy as np
import pytest

from services.card_crop_yolov8.detector import FieldDetector


class FakeBoxes:
    def __init__(self, cls, xyxy, conf):
        self.cls = np.array(cls)
        self.xyxy = np.array(xyxy)
        self.conf = np.array(conf)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, names):
        self.names = names

    def predict(self, image_bgr, conf=0.25, verbose=False):
        boxes = FakeBoxes(
            cls=[0, 1, 2],
            xyxy=[[10, 20, 100, 50], [10, 60, 100, 90], [10, 100, 100, 130]],
            conf=[0.95, 0.88, 0.72],
        )
        return [FakeResult(boxes)]


def _make_detector(names, field_map=None):
    det = FieldDetector.__new__(FieldDetector)
    det.model = FakeModel(names)
    det.conf = 0.25
    det.field_map = field_map or {}
    return det


class TestFieldMapping:
    def test_no_mapping_lowercases(self):
        det = _make_detector({0: "AADHAR_NUMBER", 1: "NAME", 2: "DATE_OF_BIRTH"})
        fields = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        labels = [f["field"] for f in fields]
        assert labels == ["aadhar_number", "name", "date_of_birth"]

    def test_explicit_mapping(self):
        field_map = {
            "AADHAR_NUMBER": "aadhaar_number",
            "NAME": "name",
            "DATE_OF_BIRTH": "dob",
        }
        det = _make_detector(
            {0: "AADHAR_NUMBER", 1: "NAME", 2: "DATE_OF_BIRTH"},
            field_map=field_map,
        )
        fields = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        labels = [f["field"] for f in fields]
        assert labels == ["aadhaar_number", "name", "dob"]

    def test_unmapped_labels_dropped_when_field_map_set(self):
        field_map = {"AADHAR_NUMBER": "aadhaar_number"}
        det = _make_detector(
            {0: "AADHAR_NUMBER", 1: "NAME", 2: "UNKNOWN_FIELD"},
            field_map=field_map,
        )
        fields = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        assert len(fields) == 1
        assert fields[0]["field"] == "aadhaar_number"

    def test_non_text_fields_dropped(self):
        field_map = {
            "NAME": "name",
            "PHOTO": "photo",
            "SIGNATURE": "signature",
        }
        det = _make_detector(
            {0: "NAME", 1: "PHOTO", 2: "SIGNATURE"},
            field_map=field_map,
        )
        fields = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        assert len(fields) == 1
        assert fields[0]["field"] == "name"

    def test_bbox_and_conf_preserved(self):
        det = _make_detector({0: "A", 1: "B", 2: "C"})
        fields = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        assert fields[0]["bbox"] == (10, 20, 100, 50)
        assert fields[0]["conf"] == pytest.approx(0.95)
        assert fields[2]["conf"] == pytest.approx(0.72)

    def test_empty_result(self):
        det = _make_detector({0: "A"})
        det.model = type("M", (), {
            "names": {0: "A"},
            "predict": lambda self, *a, **k: [FakeResult(None)],
        })()
        fields = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        assert fields == []


class TestNullDetector:
    def test_null_detector_returns_empty(self):
        from apps.workers.pipeline_loader import _NullDetector
        det = _NullDetector()
        fields = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        assert fields == []

    def test_null_detector_forces_vlm_path(self):
        from apps.workers.pipeline_loader import _NullDetector
        det = _NullDetector()
        assert det.detect(np.zeros((640, 480, 3), dtype=np.uint8)) == []
