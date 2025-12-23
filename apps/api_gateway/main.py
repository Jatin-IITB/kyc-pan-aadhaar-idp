# apps/api_gateway/main.py
from __future__ import annotations

from pathlib import Path

from services.card_crop_yolov8.detector import FieldDetector
from services.doc_classifier.classifier import DocClassifier
from services.ocr_paddle.roi_ocr import ROIOCR
from services.pipeline import KYCPipeline, PipelineConfig

from apps.api_gateway.app_factory import create_app


repo_root = Path(__file__).resolve().parents[2]
PAN_WEIGHTS = repo_root / "models/yolov8/pan_field_detector_v1/best.pt"
AADHAAR_WEIGHTS = repo_root / "models/yolov8/aadhar_field_detector_v1/best.pt"

pan_detector = FieldDetector(str(PAN_WEIGHTS), conf=0.25)
aadhaar_detector = FieldDetector(str(AADHAAR_WEIGHTS), conf=0.25)
ocr = ROIOCR(lang="en")

doc_classifier = DocClassifier(
    pan_detector=pan_detector,
    aadhaar_detector=aadhaar_detector,
    conf_threshold=0.25,
    min_best_score=1.30,
    min_margin=0.20,
    wrong_field_penalty=0.35,
)

pipeline = KYCPipeline(
    pan_detector=pan_detector,
    aadhaar_detector=aadhaar_detector,
    ocr=ocr,
    doc_classifier=doc_classifier,
    config=PipelineConfig(
        base_rotations=("rot0", "rot180", "rot90", "rot270"),
        accept_partial_score=1.75,
        accept_partial_coverage=0.75,
    ),
)

app = create_app(pipeline=pipeline, max_concurrency=4)
