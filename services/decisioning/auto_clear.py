from __future__ import annotations

from typing import Any, Dict, List

from services.decisioning.thresholds import SPOOF_AUTO_CLEAR_OVERRIDE


class AutoClearEngine:
    """Hard override rules that bypass confidence calibration."""

    def evaluate(
        self,
        calibration: Dict[str, Any],
        spoof_score: float = 0.0,
        policy_compliant: bool = True,
        critical_contradictions: int = 0,
        quality_passed: bool = True,
    ) -> Dict[str, Any]:
        overrides: List[Dict[str, str]] = []
        recommendation = calibration.get("recommendation", "REVIEW")

        if spoof_score > SPOOF_AUTO_CLEAR_OVERRIDE:
            overrides.append({"rule": "spoof_threshold", "action": "REJECT", "reason": f"spoof_score={spoof_score:.2f} > {SPOOF_AUTO_CLEAR_OVERRIDE}"})
            recommendation = "REJECT"

        if not policy_compliant:
            overrides.append({"rule": "policy_non_compliant", "action": "REVIEW", "reason": "Policy requirements not met"})
            if recommendation == "AUTO_CLEAR":
                recommendation = "REVIEW"

        if critical_contradictions > 0:
            overrides.append({"rule": "critical_contradictions", "action": "REVIEW", "reason": f"{critical_contradictions} critical contradictions"})
            if recommendation == "AUTO_CLEAR":
                recommendation = "REVIEW"

        if not quality_passed:
            overrides.append({"rule": "quality_failed", "action": "REJECT", "reason": "Image quality below threshold"})
            recommendation = "REJECT"

        return {
            "original_recommendation": calibration.get("recommendation", "REVIEW"),
            "final_recommendation": recommendation,
            "overrides_applied": overrides,
            "calibrated_confidence": calibration.get("calibrated_confidence", 0.0),
            "auto_cleared": recommendation == "AUTO_CLEAR",
        }
