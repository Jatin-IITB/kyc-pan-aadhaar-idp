from __future__ import annotations

from typing import Any, Dict, Tuple


class ExtractionEnsemble:
    """Pick the best extraction from YOLO and VLM paths, or merge at field level."""

    def __init__(self, yolo_weight: float = 0.6, vlm_weight: float = 0.4) -> None:
        self.yolo_weight = yolo_weight
        self.vlm_weight = vlm_weight

    def pick_best(
        self,
        yolo_extraction: Dict[str, Any],
        yolo_score: float,
        vlm_extraction: Dict[str, Any],
        vlm_score: float,
        doc_type: str,
    ) -> Tuple[Dict[str, Any], str, float]:
        """Returns (chosen_extraction, path_name, final_score)."""
        yolo_valid = bool(yolo_extraction)
        vlm_valid = bool(vlm_extraction)

        if yolo_valid and vlm_valid:
            weighted_yolo = yolo_score * self.yolo_weight
            weighted_vlm = vlm_score * self.vlm_weight

            if weighted_yolo >= weighted_vlm:
                merged = self.merge_field_level(yolo_extraction, vlm_extraction)
                return merged, "ensemble", max(yolo_score, vlm_score)
            else:
                merged = self.merge_field_level(vlm_extraction, yolo_extraction)
                return merged, "ensemble", max(yolo_score, vlm_score)

        if yolo_valid:
            return yolo_extraction, "yolo", yolo_score

        if vlm_valid:
            return vlm_extraction, "vlm", vlm_score

        return yolo_extraction or {}, "yolo", yolo_score

    def merge_field_level(
        self,
        primary: Dict[str, Any],
        secondary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Per-field best pick. Primary wins ties; secondary fills gaps."""
        merged = dict(primary)

        for key, sec_val in secondary.items():
            if key not in merged:
                merged[key] = sec_val
                continue

            pri_val = merged[key]
            if not isinstance(pri_val, dict) or not isinstance(sec_val, dict):
                continue

            pri_conf = 0.5 * float(pri_val.get("det_conf", 0)) + 0.5 * float(pri_val.get("ocr_conf", 0))
            sec_conf = 0.5 * float(sec_val.get("det_conf", 0)) + 0.5 * float(sec_val.get("ocr_conf", 0))

            pri_value = pri_val.get("value", "")
            sec_value = sec_val.get("value", "")

            if not pri_value and sec_value:
                merged[key] = sec_val
            elif sec_conf > pri_conf and sec_value:
                merged[key] = sec_val

        return merged
