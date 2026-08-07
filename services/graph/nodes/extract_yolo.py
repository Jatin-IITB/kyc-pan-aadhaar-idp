from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)

from services.doc_classifier.classifier import rotate_bgr
from services.extraction.normalize import normalize_extraction
from services.graph.deps import get_deps
from services.graph.state import CaseState
from services.pipeline import KYCPipeline
from services.validation.schema_validation import get_required_fields, validate_with_schema


def _collapse_best_per_field(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return KYCPipeline._collapse_best_per_field(results)


def _to_flat(extraction: Dict[str, Any]) -> Dict[str, Any]:
    return KYCPipeline._to_flat(extraction)


def _score_candidate(
    extraction: Dict[str, Any], doc_type: str
) -> Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    extraction_norm = normalize_extraction(extraction, doc_type)
    flat = _to_flat(extraction_norm)

    required = get_required_fields(doc_type)
    present_required = sum(
        1 for k in required if k in flat and str(flat[k]).strip() != ""
    )
    coverage = (present_required / max(1, len(required))) if required else 0.0

    confs: List[float] = []
    for v in extraction_norm.values():
        confs.append(
            0.5 * float(v.get("det_conf", 0.0)) + 0.5 * float(v.get("ocr_conf", 0.0))
        )
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

    is_valid, msg = validate_with_schema(flat, doc_type)
    total = (2.0 if is_valid else 0.0) + (1.0 * coverage) + (1.0 * avg_conf)

    meta = {
        "is_valid": is_valid,
        "message": msg,
        "coverage": coverage,
        "present_required": present_required,
        "avg_conf": avg_conf,
        "score": total,
    }
    return float(total), meta, flat, extraction_norm


def _run_once(img_bgr: np.ndarray, doc_type: str) -> Dict[str, Any]:
    deps = get_deps()
    detector = deps.pan_detector if doc_type == "pan" else deps.aadhaar_detector
    fields = detector.detect(img_bgr)
    results = deps.ocr.extract(img_bgr, fields)
    return _collapse_best_per_field(results)


def _best_by_schema(
    img_bgr: np.ndarray, dt: str, rotations: List[str]
) -> Tuple[float, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    best = None
    for rot in rotations:
        img_r = rotate_bgr(img_bgr, rot)
        extraction_raw = _run_once(img_r, dt)
        score, meta, flat, extraction_norm = _score_candidate(extraction_raw, dt)
        cand = (score, rot, extraction_norm, meta, flat)
        if best is None or cand[0] > best[0]:
            best = cand
        if meta.get("is_valid"):
            return cand
    if best is None:
        raise RuntimeError("No rotations evaluated.")
    return best


def extract_yolo_node(state: CaseState) -> CaseState:
    deps = get_deps()
    dt = state["doc_type"]
    img = state["image_bgr"]
    hint = state.get("rotation_hint", "rot0")
    base_rots = list(deps.config.base_rotations)

    if dt not in ("pan", "aadhaar"):
        # unknown classifier — try both doc types, pick best
        if dt == "unknown":
            pan_best = _best_by_schema(img, "pan", base_rots)
            aad_best = _best_by_schema(img, "aadhaar", base_rots)
            if pan_best[0] >= aad_best[0]:
                chosen_dt = "pan"
                score, rot, extraction_norm, meta, flat = pan_best
            else:
                chosen_dt = "aadhaar"
                score, rot, extraction_norm, meta, flat = aad_best
            return {
                **state,
                "doc_type": chosen_dt,
                "yolo_extraction": extraction_norm,
                "yolo_confidence": score,
                "extraction_normalized": extraction_norm,
                "flat_fields": flat,
                "schema_valid": meta["is_valid"],
                "validation_message": meta["message"],
                "validation_score": score,
                "rotation_hint": rot,
                "extraction_path": "yolo",
            }
        # doc type without YOLO model — skip, let VLM handle
        return {
            **state,
            "yolo_extraction": {},
            "yolo_confidence": 0.0,
        }

    rotations = [hint] + [r for r in base_rots if r != hint]
    best = None
    for rot in rotations:
        img_r = rotate_bgr(img, rot)
        extraction_raw = _run_once(img_r, dt)
        score, meta, flat, extraction_norm = _score_candidate(extraction_raw, dt)
        cand = (score, rot, extraction_norm, meta, flat)
        if best is None or cand[0] > best[0]:
            best = cand
        if meta["is_valid"]:
            best = cand
            break
        if (
            meta["coverage"] >= deps.config.accept_partial_coverage
            and meta["score"] >= deps.config.accept_partial_score
        ):
            best = cand
            break

    score, rot, extraction_norm, meta, flat = best  # type: ignore[misc]

    return {
        **state,
        "yolo_extraction": extraction_norm,
        "yolo_confidence": score,
        "extraction_normalized": extraction_norm,
        "flat_fields": flat,
        "schema_valid": meta["is_valid"],
        "validation_message": meta["message"],
        "validation_score": score,
        "rotation_hint": rot,
        "extraction_path": "yolo",
    }
