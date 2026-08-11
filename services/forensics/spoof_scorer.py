from __future__ import annotations

from typing import Any, Dict, List


class SpoofScorer:
    """Aggregate forensic signals into a single spoof score.

    Combination is **noisy-OR over gated per-detector scores**, not a weighted
    average (ADR-024). A forgery typically trips a single detector hard; a
    weighted average multiplied that signal by a small weight and diluted it
    below the flag threshold (measured 26% tamper recall on the Phase 11 red
    team). Each detector now contributes only when its own decision gate fires,
    and the gated scores combine as ``1 - prod(1 - s_i)`` so:

    - one strong detector alone can flag the document, and
    - genuine documents (all gates closed) score exactly 0.

    ``priors`` scales each detector's evidential strength — how much a positive
    verdict from that detector should weigh toward "forged".
    """

    # Font is corroboration-only (prior < the 0.2 flag threshold): stroke-width
    # analysis has no separating power on legitimately multi-font ID cards
    # (measured genuine 0.22-0.71 vs attack 0.20-0.71, fully overlapping), so it
    # must never flag a genuine document on its own. It reinforces other signals
    # via noisy-OR. True font forensics needs OCR field-region context (W5).
    DEFAULT_PRIORS = {
        "ela": 0.55,
        "copy_move": 0.90,
        "font": 0.18,
        "metadata": 0.55,
        "screen": 0.65,
    }

    def __init__(self, priors: Dict[str, float] | None = None) -> None:
        self.priors = priors or self.DEFAULT_PRIORS

    def compute(
        self,
        ela_result: Dict[str, Any],
        copy_move_result: Dict[str, Any],
        font_result: Dict[str, Any],
        metadata_result: Dict[str, Any],
        screen_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence: List[Dict[str, Any]] = []
        gated: Dict[str, float] = {k: 0.0 for k in self.DEFAULT_PRIORS}

        # --- ELA: gate on suspicious regions (recompression seams) ---
        if ela_result.get("suspicious_regions"):
            strength = min(1.0, 0.7 + ela_result.get("ela_score", 0.0) * 10)
            gated["ela"] = self.priors["ela"] * strength
            evidence.append({
                "type": "ela",
                "score": gated["ela"],
                "detail": f"{len(ela_result['suspicious_regions'])} recompression seam(s)",
            })

        # --- Copy-move: gate on detection; confidence scales strength ---
        if copy_move_result.get("detected"):
            strength = max(0.7, copy_move_result.get("confidence", 0.0))
            gated["copy_move"] = self.priors["copy_move"] * strength
            shift = copy_move_result.get("dominant_shift")
            detail = f"{len(copy_move_result.get('matched_pairs', []))} matched pair(s)"
            if shift:
                detail += f", shift {tuple(shift)}"
            evidence.append({"type": "copy_move", "score": gated["copy_move"], "detail": detail})

        # --- Font: gate on inconsistent regions ---
        inconsistent = font_result.get("inconsistent_regions") or []
        if inconsistent:
            consistency = font_result.get("font_consistency_score", 1.0)
            strength = min(1.0, 0.7 + (1.0 - consistency) * 0.6)
            gated["font"] = self.priors["font"] * strength
            evidence.append({
                "type": "font_inconsistency",
                "score": gated["font"],
                "detail": f"{len(inconsistent)} inconsistent region(s)",
            })

        # --- Metadata: gate on editor software or date anomaly ---
        if metadata_result.get("software_edited"):
            gated["metadata"] = self.priors["metadata"]
            evidence.append({
                "type": "metadata_edit",
                "score": gated["metadata"],
                "detail": "Edited with image-editing software",
            })
        elif "date_mismatch" in (metadata_result.get("metadata_flags") or []):
            gated["metadata"] = self.priors["metadata"] * 0.6
            evidence.append({
                "type": "metadata_anomaly",
                "score": gated["metadata"],
                "detail": "EXIF date mismatch",
            })

        # --- Screen recapture: gate on Moire verdict ---
        if screen_result.get("is_recaptured"):
            strength = max(0.7, screen_result.get("moire_score", 0.0))
            gated["screen"] = self.priors["screen"] * strength
            evidence.append({
                "type": "screen_recapture",
                "score": gated["screen"],
                "detail": "Moire pattern detected (photo of screen)",
            })

        # --- Noisy-OR combination of independent evidence ---
        prod = 1.0
        for s in gated.values():
            prod *= (1.0 - min(max(s, 0.0), 0.999))
        spoof_score = 1.0 - prod

        if spoof_score >= 0.7:
            risk_level, recommendation = "CRITICAL", "REJECT"
        elif spoof_score >= 0.4:
            risk_level, recommendation = "HIGH", "REVIEW"
        elif spoof_score >= 0.2:
            risk_level, recommendation = "MEDIUM", "REVIEW"
        else:
            risk_level, recommendation = "LOW", "PASS"

        return {
            "spoof_score": spoof_score,
            "risk_level": risk_level,
            "evidence": evidence,
            "recommendation": recommendation,
            "component_scores": gated,
        }
