from __future__ import annotations

from services.graph.state import CaseState
from services.preprocessing.quality import check_image_quality


def quality_gate_node(state: CaseState) -> CaseState:
    is_good, quality_meta = check_image_quality(state["image_bgr"])

    attempt_rescue = False
    if not is_good:
        reason = quality_meta.get("rejection_reason", "")
        if "overexposed" in reason or "dark" in reason:
            attempt_rescue = True

    return {
        "quality_passed": is_good or attempt_rescue,
        "quality_meta": quality_meta,
        "attempt_rescue": attempt_rescue,
    }
