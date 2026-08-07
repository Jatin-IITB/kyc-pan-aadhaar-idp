from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

import numpy as np


class CaseState(TypedDict, total=False):
    # Input
    case_id: str
    image_bgr: np.ndarray
    requested_doc_type: str

    # Quality gate
    quality_passed: bool
    quality_meta: Dict[str, Any]
    attempt_rescue: bool

    # Classification
    doc_type: str
    rotation_hint: str
    classifier_info: Optional[Dict[str, Any]]

    # Extraction — YOLO path
    yolo_extraction: Dict[str, Any]
    yolo_confidence: float

    # Extraction — VLM path (Phase 2)
    vlm_extraction: Dict[str, Any]
    vlm_confidence: float

    # Ensemble output
    chosen_extraction: Dict[str, Any]
    extraction_path: str  # "yolo", "vlm", "ensemble"

    # Validation
    extraction_normalized: Dict[str, Any]
    flat_fields: Dict[str, Any]
    schema_valid: bool
    validation_message: str
    validation_score: float

    # LLM rescue
    llm_rescue_result: Dict[str, Any]

    # Forensics (Phase 4)
    forensics_result: Dict[str, Any]
    spoof_score: float

    # Cross-doc (Phase 6)
    packet_documents: List[Dict[str, Any]]
    cross_doc_result: Dict[str, Any]

    # Policy (Phase 5)
    policy_result: Dict[str, Any]

    # Calibration (Phase 7)
    calibration_result: Dict[str, Any]

    # Decision
    decision: str
    final_confidence: float
    final_result: Dict[str, Any]

    # Audit (Phase 7)
    audit_events: List[Dict[str, Any]]
