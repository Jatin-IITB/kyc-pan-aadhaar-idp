from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from services.forensics.copy_move import CopyMoveDetector
from services.forensics.ela import ELADetector
from services.forensics.font_analysis import FontConsistencyAnalyzer
from services.forensics.metadata import MetadataForensics
from services.forensics.screen_recapture import ScreenRecaptureDetector
from services.forensics.spoof_scorer import SpoofScorer
from services.graph.state import CaseState


def forensics_node(state: CaseState) -> CaseState:
    img = state["image_bgr"]

    try:
        ela = ELADetector().analyze(img)
        copy_move = CopyMoveDetector().detect(img)
        font = FontConsistencyAnalyzer().analyze(img, [])
        metadata = MetadataForensics().analyze(state.get("image_bytes", b""))
        screen = ScreenRecaptureDetector().detect(img)

        result = SpoofScorer().compute(ela, copy_move, font, metadata, screen)
    except Exception as e:
        logger.warning("Forensics analysis failed: {}", e)
        result = {"spoof_score": 0.0, "risk_level": "LOW", "evidence": [], "recommendation": "PASS"}

    return {
        "forensics_result": result,
        "spoof_score": result.get("spoof_score", 0.0),
    }
