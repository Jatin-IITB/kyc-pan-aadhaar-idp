from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from functools import lru_cache

from services.forensics.copy_move import CopyMoveDetector
from services.forensics.ela import ELADetector
from services.forensics.font_analysis import FontConsistencyAnalyzer
from services.forensics.font_profile import TemplateFontForensics
from services.forensics.metadata import MetadataForensics
from services.forensics.screen_recapture import ScreenRecaptureDetector
from services.forensics.spoof_scorer import SpoofScorer
from services.graph.state import CaseState


@lru_cache(maxsize=1)
def _template_font() -> TemplateFontForensics:
    """Profile JSON is read once; the detector itself is stateless."""
    return TemplateFontForensics()


def forensics_node(state: CaseState) -> CaseState:
    img = state["image_bgr"]

    try:
        ela = ELADetector().analyze(img)
        copy_move = CopyMoveDetector().detect(img)
        font = FontConsistencyAnalyzer().analyze(img, [])
        metadata = MetadataForensics().analyze(state.get("image_bytes", b""))
        screen = ScreenRecaptureDetector().detect(img)

        # Needs the doc type to pick the right calibrated envelope; classify
        # runs before forensics in the graph so this is populated.
        font_template = _template_font().analyze(img, state.get("doc_type", ""))

        result = SpoofScorer().compute(ela, copy_move, font, metadata, screen,
                                       font_template_result=font_template)
    except Exception as e:
        logger.warning("Forensics analysis failed: {}", e)
        result = {"spoof_score": 0.0, "risk_level": "LOW", "evidence": [], "recommendation": "PASS"}

    return {
        "forensics_result": result,
        "spoof_score": result.get("spoof_score", 0.0),
    }
