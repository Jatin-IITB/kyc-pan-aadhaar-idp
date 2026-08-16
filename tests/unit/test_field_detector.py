"""Tests for FieldDetector field name mapping and _NullDetector."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.card_crop_yolov8.detector import FieldDetector

REPO_ROOT = Path(__file__).resolve().parents[2]


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


class TestFieldMapCoversInstalledModels:
    """Guards against field_map / model-class drift.

    A model whose classes aren't in field_map silently yields ZERO fields —
    FieldDetector drops unmapped labels by design. That failure is invisible
    at runtime (the pipeline just falls back to VLM), so pin it here.

    Skips when weights aren't present so CI stays green without models.
    """

    @staticmethod
    def _cases():
        import yaml
        cfg = yaml.safe_load(
            (REPO_ROOT / "config" / "models.yaml").read_text()
        )["yolov8"]
        return [(k, cfg[k]) for k in ("pan_fields", "aadhar_fields")]

    @pytest.mark.parametrize("key", ["pan_fields", "aadhar_fields"])
    def test_every_model_class_is_mapped(self, key):
        import yaml
        cfg = yaml.safe_load(
            (REPO_ROOT / "config" / "models.yaml").read_text()
        )["yolov8"][key]

        weights = REPO_ROOT / cfg["weights"]
        if not weights.exists():
            pytest.skip(f"{key} weights not present at {weights}")

        field_map = cfg.get("field_map") or {}
        assert field_map, f"{key} has no field_map — all detections would pass through raw"

        from ultralytics import YOLO
        model_classes = set(YOLO(str(weights)).names.values())
        unmapped = sorted(model_classes - set(field_map))
        assert not unmapped, (
            f"{key}: model emits classes absent from field_map in config/models.yaml: "
            f"{unmapped}. These are silently DROPPED, so the detector would yield "
            f"no fields for them."
        )


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
