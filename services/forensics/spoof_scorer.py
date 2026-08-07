from __future__ import annotations

from typing import Any, Dict, List


class SpoofScorer:
    """Weighted aggregate of all forensics signals into a single spoof score."""

    DEFAULT_WEIGHTS = {
        "ela": 0.25,
        "copy_move": 0.25,
        "font": 0.15,
        "metadata": 0.15,
        "screen": 0.20,
    }

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS

    def compute(
        self,
        ela_result: Dict[str, Any],
        copy_move_result: Dict[str, Any],
        font_result: Dict[str, Any],
        metadata_result: Dict[str, Any],
        screen_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence: List[Dict[str, Any]] = []

        ela_score = min(1.0, ela_result.get("ela_score", 0.0) * 10)
        if ela_result.get("suspicious_regions"):
            ela_score = max(ela_score, 0.5)
            evidence.append({
                "type": "ela",
                "score": ela_score,
                "detail": f"{len(ela_result['suspicious_regions'])} suspicious region(s)",
            })

        cm_score = copy_move_result.get("confidence", 0.0)
        if copy_move_result.get("detected"):
            cm_score = max(cm_score, 0.6)
            evidence.append({
                "type": "copy_move",
                "score": cm_score,
                "detail": f"{len(copy_move_result.get('matched_pairs', []))} matched pair(s)",
            })

        font_score = max(0.0, 1.0 - font_result.get("font_consistency_score", 1.0))
        if font_result.get("inconsistent_regions"):
            font_score = max(font_score, 0.4)
            evidence.append({
                "type": "font_inconsistency",
                "score": font_score,
                "detail": f"{len(font_result['inconsistent_regions'])} inconsistent region(s)",
            })

        meta_score = 0.0
        if metadata_result.get("software_edited"):
            meta_score = 0.7
            evidence.append({
                "type": "metadata_edit",
                "score": meta_score,
                "detail": "Edited with image editing software",
            })
        elif metadata_result.get("metadata_flags"):
            flags = metadata_result["metadata_flags"]
            if "date_mismatch" in flags:
                meta_score = 0.4
                evidence.append({
                    "type": "metadata_anomaly",
                    "score": meta_score,
                    "detail": "Date mismatch in EXIF",
                })

        screen_score = screen_result.get("moire_score", 0.0)
        if screen_result.get("is_recaptured"):
            screen_score = max(screen_score, 0.6)
            evidence.append({
                "type": "screen_recapture",
                "score": screen_score,
                "detail": "Moire pattern detected (photo of screen)",
            })

        spoof_score = (
            self.weights["ela"] * ela_score
            + self.weights["copy_move"] * cm_score
            + self.weights["font"] * font_score
            + self.weights["metadata"] * meta_score
            + self.weights["screen"] * screen_score
        )
        spoof_score = min(1.0, spoof_score)

        if spoof_score >= 0.7:
            risk_level = "CRITICAL"
            recommendation = "REJECT"
        elif spoof_score >= 0.4:
            risk_level = "HIGH"
            recommendation = "REVIEW"
        elif spoof_score >= 0.2:
            risk_level = "MEDIUM"
            recommendation = "REVIEW"
        else:
            risk_level = "LOW"
            recommendation = "PASS"

        return {
            "spoof_score": spoof_score,
            "risk_level": risk_level,
            "evidence": evidence,
            "recommendation": recommendation,
            "component_scores": {
                "ela": ela_score,
                "copy_move": cm_score,
                "font": font_score,
                "metadata": meta_score,
                "screen": screen_score,
            },
        }
