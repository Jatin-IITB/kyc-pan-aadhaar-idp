# services/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.extraction.normalize import normalize_extraction
from services.validation.schema_validation import validate_with_schema, get_required_fields
from services.doc_classifier.classifier import DocClassifier, rotate_bgr


class PipelineError(RuntimeError):
    """Non-HTTP error for pipeline failures."""


@dataclass(frozen=True)
class PipelineConfig:
    base_rotations: Tuple[str, ...] = ("rot0", "rot180", "rot90", "rot270")
    accept_partial_score: float = 1.75
    accept_partial_coverage: float = 0.75


class KYCPipeline:
    def __init__(
        self,
        *,
        pan_detector: Any,
        aadhaar_detector: Any,
        ocr: Any,
        doc_classifier: DocClassifier,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.pan_detector = pan_detector
        self.aadhaar_detector = aadhaar_detector
        self.ocr = ocr
        self.doc_classifier = doc_classifier
        self.config = config or PipelineConfig()

    @staticmethod
    def _normalize_doc_type(doc_type: str) -> str:
        dt = (doc_type or "auto").lower().strip()
        return "aadhaar" if dt == "aadhar" else dt

    @staticmethod
    def _normalize_key(field_name: str) -> str:
        k = field_name.lower().strip()
        mapping = {
            "aadhaar_number": "aadhaar_number",
            "aadhar_number": "aadhaar_number",
            "dob": "date_of_birth",
            "date_of_birth": "date_of_birth",
        }
        return mapping.get(k, k)

    @classmethod
    def _collapse_best_per_field(cls, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Pick best candidate per normalized key by average(det_conf, ocr_conf).
        """
        best: Dict[str, Dict[str, Any]] = {}
        for r in results:
            key = cls._normalize_key(r["field"])
            score = 0.5 * float(r["det_conf"]) + 0.5 * float(r["ocr_conf"])
            if key not in best or score > float(best[key]["_score"]):
                best[key] = {**r, "_score": score, "_key": key}

        extraction: Dict[str, Dict[str, Any]] = {}
        for v in best.values():
            extraction[v["_key"]] = {
                "value": v["text"],
                "det_conf": float(v["det_conf"]),
                "ocr_conf": float(v["ocr_conf"]),
                "bbox": v.get("bbox"),
            }
        return extraction

    @staticmethod
    def _to_flat(extraction: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: (v.get("value") if isinstance(v, dict) else v)
            for k, v in (extraction or {}).items()
        }

    def score_candidate(
        self, extraction: Dict[str, Any], doc_type: str
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Returns: (score, meta, flat, extraction_norm)
        """
        extraction_norm = normalize_extraction(extraction, doc_type)
        flat = self._to_flat(extraction_norm)

        required = get_required_fields(doc_type)
        present_required = sum(1 for k in required if k in flat and str(flat[k]).strip() != "")
        coverage = (present_required / max(1, len(required))) if required else 0.0

        confs: List[float] = []
        for v in extraction_norm.values():
            confs.append(0.5 * float(v["det_conf"]) + 0.5 * float(v["ocr_conf"]))
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

        is_valid, msg = validate_with_schema(flat, doc_type)
        total = (2.0 if is_valid else 0.0) + (1.0 * coverage) + (1.0 * avg_conf)

        meta = {
            "is_valid": is_valid,
            "message": msg,
            "coverage": coverage,
            "present_required": present_required,
            "required_total": len(required),
            "avg_conf": avg_conf,
            "score": total,
        }
        return float(total), meta, flat, extraction_norm

    def run_pipeline_once(self, img_bgr: np.ndarray, doc_type: str) -> Dict[str, Any]:
        detector = self.pan_detector if doc_type == "pan" else self.aadhaar_detector
        fields = detector.detect(img_bgr)
        results = self.ocr.extract(img_bgr, fields)
        return self._collapse_best_per_field(results)

    def best_by_schema(self, img_bgr: np.ndarray, dt: str, rotations: List[str]) -> Tuple[float, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        best: Optional[Tuple[float, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = None
        for rot in rotations:
            img_r = rotate_bgr(img_bgr, rot)
            extraction = self.run_pipeline_once(img_r, dt)
            score, meta, flat, _ = self.score_candidate(extraction, dt)
            cand = (score, rot, extraction, meta, flat)
            if best is None or cand[0] > best[0]:
                best = cand
            if meta["is_valid"]:
                return cand

        if best is None:
            raise PipelineError("No rotations evaluated.")
        return best

    def extract_from_bgr(self, img_bgr: np.ndarray, doc_type: str) -> Dict[str, Any]:
        dt = self._normalize_doc_type(doc_type)
        base_rots = list(self.config.base_rotations)

        clf_info = None
        chosen_rotation_hint = "rot0"

        # AUTO ROUTING (detection-only)
        if dt in ["auto", ""]:
            route = self.doc_classifier.predict(img_bgr)
            clf_info = {
                "doc_type": route.doc_type,
                "rotation": route.rotation,
                "best_score": route.best_score,
                "score_pan": route.score_pan,
                "score_aadhaar": route.score_aadhaar,
                "margin": route.margin,
                "confident": route.confident,
                "details": route.details,
            }
            chosen_rotation_hint = route.rotation

            if route.doc_type == "unknown":
                pan_best = self.best_by_schema(img_bgr, "pan", base_rots)
                aad_best = self.best_by_schema(img_bgr, "aadhaar", base_rots)

                if pan_best[0] >= aad_best[0]:
                    dt = "pan"
                    _, rot, extraction, meta, _ = pan_best
                else:
                    dt = "aadhaar"
                    _, rot, extraction, meta, _ = aad_best

                return {
                    "document_type": dt,
                    "chosen_rotation": rot,
                    "classifier": clf_info,
                    "extraction": extraction,
                    "validation": {"is_valid": meta["is_valid"], "message": meta["message"]},
                    "selection": meta,
                    "routing_mode": "fallback_schema",
                }

            dt = route.doc_type

        if dt not in ["pan", "aadhaar"]:
            raise PipelineError("Invalid doc_type. Use 'pan', 'aadhaar', or 'auto'.")

        # ROTATION SEARCH (schema-guided) for the chosen doc type
        rotations = [chosen_rotation_hint] + [r for r in base_rots if r != chosen_rotation_hint]

        best: Optional[Tuple[float, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = None
        for rot in rotations:
            img_r = rotate_bgr(img_bgr, rot)
            extraction = self.run_pipeline_once(img_r, dt)
            score, meta, flat, _ = self.score_candidate(extraction, dt)
            cand = (score, rot, extraction, meta, flat)

            if best is None or cand[0] > best[0]:
                best = cand

            if meta["is_valid"]:
                return {
                    "document_type": dt,
                    "chosen_rotation": rot,
                    "classifier": clf_info,
                    "extraction": extraction,
                    "validation": {"is_valid": True, "message": "Valid"},
                    "selection": meta,
                    "routing_mode": "detector_then_schema",
                }

            if meta["coverage"] >= self.config.accept_partial_coverage and meta["score"] >= self.config.accept_partial_score:
                return {
                    "document_type": dt,
                    "chosen_rotation": rot,
                    "classifier": clf_info,
                    "extraction": extraction,
                    "validation": {"is_valid": False, "message": meta["message"]},
                    "selection": meta,
                    "routing_mode": "detector_then_schema",
                }

        if best is None:
            raise PipelineError("No rotations evaluated.")

        _, rot, extraction, meta, _ = best
        return {
            "document_type": dt,
            "chosen_rotation": rot,
            "classifier": clf_info,
            "extraction": extraction,
            "validation": {"is_valid": meta["is_valid"], "message": meta["message"]},
            "selection": meta,
            "routing_mode": "detector_then_schema",
        }
