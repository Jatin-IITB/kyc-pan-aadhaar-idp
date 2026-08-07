from __future__ import annotations

import math
from typing import Any, Dict, Optional


class ConfidenceCalibrator:
    """Weighted confidence aggregation with temperature scaling for auto-clear decisions."""

    WEIGHTS = {
        "extraction": 0.35,
        "forensics": 0.25,
        "policy": 0.25,
        "cross_doc": 0.15,
    }

    AUTO_CLEAR_THRESHOLD = 0.92
    REVIEW_THRESHOLD = 0.70

    def __init__(self, temperature: float = 1.5) -> None:
        self.temperature = temperature

    def _apply_temperature(self, score: float) -> float:
        if score <= 0.0:
            return 0.0
        if score >= 1.0:
            return 1.0
        logit = math.log(score / (1.0 - score))
        scaled = logit / self.temperature
        return 1.0 / (1.0 + math.exp(-scaled))

    def calibrate(
        self,
        extraction_score: float = 0.0,
        forensics_score: float = 1.0,
        policy_score: float = 1.0,
        cross_doc_score: float = 1.0,
    ) -> Dict[str, Any]:
        raw_scores = {
            "extraction": max(0.0, min(1.0, extraction_score)),
            "forensics": max(0.0, min(1.0, 1.0 - forensics_score)),
            "policy": max(0.0, min(1.0, policy_score)),
            "cross_doc": max(0.0, min(1.0, cross_doc_score)),
        }

        weighted_sum = sum(raw_scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        calibrated = self._apply_temperature(weighted_sum)

        if calibrated >= self.AUTO_CLEAR_THRESHOLD:
            recommendation = "AUTO_CLEAR"
        elif calibrated >= self.REVIEW_THRESHOLD:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"

        return {
            "raw_scores": raw_scores,
            "weighted_raw": weighted_sum,
            "calibrated_confidence": calibrated,
            "recommendation": recommendation,
            "temperature": self.temperature,
        }
