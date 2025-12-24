from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from services.card_crop_yolov8.detector import FieldDetector
from services.doc_classifier.classifier import DocClassifier
from services.ocr_paddle.roi_ocr import ROIOCR
from services.pipeline import KYCPipeline, PipelineConfig


@lru_cache(maxsize=1)
def get_pipeline() -> KYCPipeline:
    repo_root = Path(__file__).resolve().parents[2]

    pan_w = repo_root / "models/yolov8/pan_field_detector_v1/best.pt"
    aad_w = repo_root / "models/yolov8/aadhar_field_detector_v1/best.pt"

    pan_detector = FieldDetector(str(pan_w), conf=0.25)
    aadhaar_detector = FieldDetector(str(aad_w), conf=0.25)

    ocr = ROIOCR(lang="en")

    doc_classifier = DocClassifier(
        pan_detector=pan_detector,
        aadhaar_detector=aadhaar_detector,
        conf_threshold=0.25,
        min_best_score=1.30,
        min_margin=0.20,
        wrong_field_penalty=0.35,
    )

    return KYCPipeline(
        pan_detector=pan_detector,
        aadhaar_detector=aadhaar_detector,
        ocr=ocr,
        doc_classifier=doc_classifier,
        config=PipelineConfig(),
    )
