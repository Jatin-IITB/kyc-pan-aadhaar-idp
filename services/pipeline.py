# services/pipeline.py
from __future__ import annotations
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

from services.extraction.normalize import normalize_extraction
from services.extraction.llm_cleaner import LLMKycCleaner
from services.validation.schema_validation import validate_with_schema, get_required_fields
from services.doc_classifier.classifier import DocClassifier, rotate_bgr
from services.preprocessing.quality import check_image_quality, resize_if_huge

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        llm_cleaner : Optional[LLMKycCleaner] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.pan_detector = pan_detector
        self.aadhaar_detector = aadhaar_detector
        self.ocr = ocr
        self.doc_classifier = doc_classifier
        self.llm_cleaner = llm_cleaner
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
        best: Dict[str, Dict[str, Any]] = {}
        for r in results:
            key = cls._normalize_key(r["field"])
            score = 0.5 * float(r.get("det_conf", 0)) + 0.5 * float(r.get("ocr_conf", 0))
            if key not in best or score > float(best[key]["_score"]):
                best[key] = {**r, "_score": score, "_key": key}

        extraction: Dict[str, Dict[str, Any]] = {}
        for v in best.values():
            extraction[v["_key"]] = {
                "value": v["text"],
                "det_conf": float(v.get("det_conf", 0)),
                "ocr_conf": float(v.get("ocr_conf", 0)),
                "bbox": v.get("bbox"),
            }
        return extraction

    @staticmethod
    def _to_flat(extraction: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: (v.get("value") if isinstance(v, dict) else v)
            for k, v in (extraction or {}).items()
        }
    _PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    _YEAR_RE = re.compile(r"(?:19|20)\d{2}")

    _CONFUSABLE = {
        "O": "0", "D": "0", "Q": "0",
        "I": "1", "L": "1", "|": "1",
        "Z": "2", "S": "5", "B": "8", "G": "6",
    }

    _CLASSES = [
        set(["0", "O", "D", "Q"]),
        set(["1", "I", "L", "|"]),
        set(["2", "Z"]),
        set(["5", "S"]),
        set(["8", "B"]),
        set(["6", "G"]),
    ]

    @staticmethod
    def _safe_str(x: Any) -> str:
        return "" if x is None else str(x).strip()

    @classmethod
    def _same_class(cls, a: str, b: str) -> bool:
        if a == b:
            return True
        for s in cls._CLASSES:
            if a in s and b in s:
                return True
        return False

    @classmethod
    def _seq_ratio(cls, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    @classmethod
    def _confusable_only_same_length(cls, a: str, b: str) -> bool:
        if len(a) != len(b) or not a:
            return False
        return all(cls._same_class(x, y) for x, y in zip(a, b))

    @classmethod
    def _canonical_pan(cls, s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", cls._safe_str(s).upper())

    @classmethod
    def _canonical_aadhaar_digits(cls, s: str) -> str:
        raw = cls._safe_str(s).upper()
        mapped = "".join(cls._CONFUSABLE.get(ch, ch) for ch in raw)
        return re.sub(r"\D", "", mapped)

    @classmethod
    def _recover_dob_or_year(cls, s: str) -> str:
        raw = cls._safe_str(s).upper()
        if not raw:
            return ""
        mapped = "".join(cls._CONFUSABLE.get(ch, ch) for ch in raw)
        mapped = mapped.replace("-", "/").replace(".", "/")
        slim = re.sub(r"[^0-9/]", "", mapped)

        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})", slim)
        if m:
            d = int(m.group(1))
            mo = int(m.group(2))
            y = m.group(3)
            if 1 <= d <= 31 and 1 <= mo <= 12:
                return f"{d:02d}/{mo:02d}/{y}"

        y = cls._YEAR_RE.search(slim)
        if y and re.fullmatch(r"(?:19|20)\d{2}", y.group(0)):
            return y.group(0)

        return ""

    @classmethod
    def _accept_pan_update(cls, orig: str, sug: str) -> bool:
        o = cls._canonical_pan(orig)
        s = cls._canonical_pan(sug)
        if len(o) != 10 or len(s) != 10:
            return False
        if not cls._PAN_RE.fullmatch(s):
            return False
        if cls._seq_ratio(o, s) < 0.85:
            return False
        return cls._confusable_only_same_length(o, s)

    @classmethod
    def _accept_aadhaar_update(cls, orig: str, sug: str) -> bool:
        o = cls._canonical_aadhaar_digits(orig)
        s = cls._canonical_aadhaar_digits(sug)
        if len(o) != 12 or len(s) != 12:
            return False
        if cls._seq_ratio(o, s) < 0.98:
            return False
        return cls._confusable_only_same_length(o, s)

    @classmethod
    def _accept_dob_update(cls, orig: str, sug: str) -> bool:
        recovered = cls._recover_dob_or_year(orig)
        if not recovered:
            return False

        sug_s = cls._safe_str(sug)
        if not sug_s:
            return False

        if re.fullmatch(r"(?:19|20)\d{2}", recovered):
            return bool(re.fullmatch(r"(?:19|20)\d{2}", sug_s)) and sug_s == recovered

        return sug_s == recovered

    @classmethod
    def _accept_name_update(cls, orig: str, sug: str) -> bool:
        o = cls._safe_str(orig)
        s = cls._safe_str(sug)
        if not o or not s:
            return False
        # allow underscore/space normalization; still keep similarity meaningful
        o2 = re.sub(r"\s+", " ", o.replace("_", " ")).strip()
        s2 = re.sub(r"\s+", " ", s.replace("_", " ")).strip()
        if not o2 or not s2:
            return False
        return cls._seq_ratio(o2.upper(), s2.upper()) >= 0.85

    def score_candidate(self, extraction: Dict[str, Any], doc_type: str) -> Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        extraction_norm = normalize_extraction(extraction, doc_type)
        flat = self._to_flat(extraction_norm)

        required = get_required_fields(doc_type)
        present_required = sum(1 for k in required if k in flat and str(flat[k]).strip() != "")
        coverage = (present_required / max(1, len(required))) if required else 0.0

        confs: List[float] = []
        for v in extraction_norm.values():
            confs.append(0.5 * float(v.get("det_conf", 0.0)) + 0.5 * float(v.get("ocr_conf", 0.0)))
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

    def run_pipeline_once(self, img_bgr: np.ndarray, doc_type: str) -> Dict[str, Any]:
        detector = self.pan_detector if doc_type == "pan" else self.aadhaar_detector
        fields = detector.detect(img_bgr)
        results = self.ocr.extract(img_bgr, fields)
        return self._collapse_best_per_field(results)

    def best_by_schema(self, img_bgr: np.ndarray, dt: str, rotations: List[str]) -> Tuple[float, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        best = None
        for rot in rotations:
            img_r = rotate_bgr(img_bgr, rot)
            extraction_raw = self.run_pipeline_once(img_r, dt)
            score, meta, flat, extraction_norm = self.score_candidate(extraction_raw, dt)
            cand = (score, rot, extraction_norm, meta, flat)
            if best is None or cand[0] > best[0]:
                best = cand
            if meta.get("is_valid"):
                return cand
        if best is None:
            raise PipelineError("No rotations evaluated.")
        return best
    
    def _extract_directed(self, img, dt, base_rots, hint, clf_info, quality_meta):
        rotations = [hint] + [r for r in base_rots if r != hint]
        best = None
        for rot in rotations:
            img_r = rotate_bgr(img, rot)
            extraction_raw = self.run_pipeline_once(img_r, dt)
            score, meta, flat, extraction_norm = self.score_candidate(extraction_raw, dt)
            cand = (score, rot, extraction_norm, meta, flat)
            
            if best is None or cand[0] > best[0]:
                best = cand
            if meta["is_valid"]:
                 return self._finalize_result(dt, rot, clf_info, extraction_norm, meta, "detector_then_schema", quality_meta)
            if meta["coverage"] >= self.config.accept_partial_coverage and meta["score"] >= self.config.accept_partial_score:
                 return self._finalize_result(dt, rot, clf_info, extraction_norm, meta, "detector_then_schema", quality_meta)
        
        _, rot, extraction, meta, _ = best
        
        # Check Back Side Content
        critical_count = 0
        if extraction.get("aadhaar_number", {}).get("value"): critical_count += 1
        if extraction.get("pan_number", {}).get("value"): critical_count += 1
        if extraction.get("name", {}).get("value"): critical_count += 1
        if extraction.get("date_of_birth", {}).get("value"): critical_count += 1
        
        if critical_count < 2:
             return self._make_reject_result(dt, quality_meta, "REJECTED_CONTENT")

        return self._finalize_result(dt, rot, clf_info, extraction, meta, "detector_then_schema", quality_meta)

    def extract_from_bgr(self, img_bgr: np.ndarray, doc_type: str) -> Dict[str, Any]:
        # 1. RESIZE
        img_bgr = resize_if_huge(img_bgr)
        
        # 2. QUALITY GATE (Conditional)
        is_good, quality_meta = check_image_quality(img_bgr)
        
        # SENIOR ENGINEER FIX:
        # If rejected for GLARE/EXPOSURE only, we perform a "Rescue Attempt".
        # We proceed to extraction anyway. If extraction is High Confidence, we Override rejection.
        # If rejected for BLUR, we trust it (Blur kills OCR).
        
        attempt_rescue = False
        if not is_good:
            reason = quality_meta.get("rejection_reason", "")
            if "overexposed" in reason or "dark" in reason:
                # Glare often doesn't hide text (black text on white card). We try to read it.
                attempt_rescue = True
            else:
                # Blur or other issues -> Hard Reject
                return self._make_reject_result(doc_type, quality_meta, "REJECTED_QUALITY")

        # 3. EXTRACTION (Proceed if Good OR Rescue Attempt)
        dt = self._normalize_doc_type(doc_type)
        base_rots = list(self.config.base_rotations)
        clf_info = None
        chosen_rotation_hint = "rot0"

        try:
            if dt in ["auto", ""]:
                route = self.doc_classifier.predict(img_bgr)
                clf_info = {"doc_type": route.doc_type, "rotation": route.rotation, "best_score": route.best_score}
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
                    final_res = self._finalize_result(dt, rot, clf_info, extraction, meta, "fallback_schema", quality_meta)
                else:
                    dt = route.doc_type
                    final_res = self._extract_directed(img_bgr, dt, base_rots, chosen_rotation_hint, clf_info, quality_meta)
            else:
                final_res = self._extract_directed(img_bgr, dt, base_rots, chosen_rotation_hint, clf_info, quality_meta)
                
        except PipelineError as e:
            # If pipeline failed completely, we can't rescue
             return self._make_reject_result(doc_type, quality_meta, "REJECTED_QUALITY")
        if not final_res["validation"]["is_valid"]:
            final_res = self._apply_llm_rescue(dt,final_res,schema_message=final_res["validation"]["message"])
        # 4. RESCUE DECISION
        if attempt_rescue:
            # If we extracted valid data, we Override the Quality Rejection
            if final_res["validation"]["is_valid"]:
                # Success! The text was readable despite glare.
                existing = final_res.get("quality_check", {}).get("rejection_reason")
                suffix = "RESCUED: Readable Content"
                final_res["quality_check"]["rejection_reason"] = f"{existing} | {suffix}" if existing else suffix
                final_res["status"] = "SUCCESS" 
                return final_res
            else:
                # Rescue failed (still invalid data). Revert to Original Quality Rejection.
                return self._make_reject_result(doc_type, quality_meta, "REJECTED_QUALITY")
        
        return final_res

    def _make_reject_result(self, dt, quality_meta, status):
        return {
            "document_type": dt or "unknown",
            "quality_check": quality_meta,
            "extraction": {},
            "validation": {"is_valid": False, "message": quality_meta.get("rejection_reason", "Rejected")},
            "status": status
        }

    def _finalize_result(self, dt, rot, clf, extraction, meta, mode, quality):
        return {
            "document_type": dt,
            "chosen_rotation": rot,
            "classifier": clf,
            "quality_check": quality,
            "extraction": extraction,
            "validation": {"is_valid": meta["is_valid"], "message": meta["message"]},
            "selection": meta,
            "routing_mode": mode,
            "status": "SUCCESS" if meta["is_valid"] else "PARTIAL_SUCCESS"
        }
    def _apply_llm_rescue(self, dt: str, final_res: Dict[str, Any], *, schema_message: str) -> Dict[str, Any]:
        if self.llm_cleaner is None:
            return final_res
        if not isinstance(final_res, dict):
            return final_res

        extraction = final_res.get("extraction") or {}
        if not isinstance(extraction, dict) or not extraction:
            return final_res

        flat_before = self._to_flat(extraction)

        try:
            suggestions = self.llm_cleaner.clean_fields(
                doc_type=dt,
                fields=flat_before,
                failure_reason=schema_message,
            )
        except Exception as e:
            logger.warning("LLM rescue failed: %s", e)
            return final_res

        if not isinstance(suggestions, dict) or not suggestions:
            return final_res

        updated: Dict[str, Any] = {}
        rejected: Dict[str, Any] = {}

        for key, suggested_val in suggestions.items():
            # Schema lock: only update keys that already exist in extraction
            if key not in extraction:
                continue

            field_obj = extraction.get(key)
            if not isinstance(field_obj, dict):
                continue

            orig_val = field_obj.get("value")
            orig_s = self._safe_str(orig_val)
            sug_s = self._safe_str(suggested_val)

            if not sug_s or sug_s == orig_s:
                continue

            accept = False
            if key == "pan_number":
                accept = self._accept_pan_update(orig_s, sug_s)
            elif key == "aadhaar_number":
                accept = self._accept_aadhaar_update(orig_s, sug_s)
            elif key == "date_of_birth":
                accept = self._accept_dob_update(orig_s, sug_s)
            elif key in ("name", "father_name"):
                accept = self._accept_name_update(orig_s, sug_s)
            else:
                accept = False

            if not accept:
                rejected[key] = {"original": orig_s, "suggested": sug_s}
                continue

            field_obj["value"] = sug_s
            meta = field_obj.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta["source"] = "llm_rescue"
                meta["conf"] = 0.80

            updated[key] = {"original": orig_s, "suggested": sug_s}

        if not updated:
            final_res.setdefault("llm_rescue", {})
            if isinstance(final_res["llm_rescue"], dict):
                final_res["llm_rescue"].update({"attempted": True, "updated": {}, "rejected": rejected})
            return final_res

        # Normalize + single re-validation at end
        extraction_norm = normalize_extraction(extraction, dt)
        flat_after = self._to_flat(extraction_norm)
        is_valid_after, msg_after = validate_with_schema(flat_after, dt)

        final_res["extraction"] = extraction_norm
        final_res.setdefault("validation", {})
        if isinstance(final_res["validation"], dict):
            final_res["validation"]["is_valid"] = bool(is_valid_after)
            final_res["validation"]["message"] = msg_after

        sel = final_res.get("selection")
        if isinstance(sel, dict):
            sel["is_valid"] = bool(is_valid_after)
            sel["message"] = msg_after

        final_res["status"] = "SUCCESS" if is_valid_after else "PARTIAL_SUCCESS"

        final_res.setdefault("llm_rescue", {})
        if isinstance(final_res["llm_rescue"], dict):
            final_res["llm_rescue"].update(
                {
                    "attempted": True,
                    "updated": updated,
                    "rejected": rejected,
                    "after": {"is_valid": bool(is_valid_after), "message": msg_after},
                }
            )

        return final_res

