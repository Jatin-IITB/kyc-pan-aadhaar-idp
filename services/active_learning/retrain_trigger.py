from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.active_learning.ground_truth_db import GroundTruthDB

logger = logging.getLogger(__name__)


class RetrainTrigger:
    """Evaluate whether retraining should be triggered based on correction volume or quality drift."""

    def __init__(
        self,
        correction_threshold: int = 100,
        f1_drop_threshold: float = 0.02,
    ) -> None:
        self.correction_threshold = correction_threshold
        self.f1_drop_threshold = f1_drop_threshold

    def should_retrain(
        self,
        ground_truth_db: GroundTruthDB,
        current_f1: Optional[float] = None,
        baseline_f1: Optional[float] = None,
    ) -> Dict[str, Any]:
        stats = ground_truth_db.get_stats()
        total_corrections = stats.get("total_corrections", 0)

        triggers = []

        if total_corrections >= self.correction_threshold:
            triggers.append({
                "reason": "correction_volume",
                "detail": f"{total_corrections} corrections >= threshold {self.correction_threshold}",
            })

        if current_f1 is not None and baseline_f1 is not None:
            drop = baseline_f1 - current_f1
            if drop >= self.f1_drop_threshold:
                triggers.append({
                    "reason": "f1_drop",
                    "detail": f"F1 dropped {drop:.4f} (baseline={baseline_f1:.4f}, current={current_f1:.4f})",
                })

        distribution = ground_truth_db.error_distribution()
        top_errors = distribution.get("top_errors", [])
        if top_errors:
            worst = top_errors[0]
            if worst["count"] >= self.correction_threshold // 2:
                triggers.append({
                    "reason": "concentrated_errors",
                    "detail": f"{worst['key']} has {worst['count']} corrections",
                })

        should = len(triggers) > 0

        if should:
            logger.info("Retrain triggered: %s", [t["reason"] for t in triggers])

        return {
            "should_retrain": should,
            "triggers": triggers,
            "total_corrections": total_corrections,
            "correction_threshold": self.correction_threshold,
        }
