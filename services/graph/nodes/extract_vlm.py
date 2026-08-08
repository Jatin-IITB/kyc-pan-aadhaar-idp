from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from services.doc_classifier.classifier import rotate_bgr
from services.graph.deps import get_deps
from services.graph.nodes.extract_yolo import _score_candidate
from services.graph.state import CaseState


def extract_vlm_node(state: CaseState) -> CaseState:
    deps = get_deps()
    if deps.vlm_extractor is None:
        return {"vlm_extraction": {}, "vlm_confidence": 0.0}

    dt = state.get("doc_type", "unknown")
    img = state["image_bgr"]
    rot = state.get("rotation_hint", "rot0")

    try:
        img_rotated = rotate_bgr(img, rot)
        extraction = deps.vlm_extractor.extract_fields(img_rotated, dt)
        if not extraction:
            return {"vlm_extraction": {}, "vlm_confidence": 0.0}

        score, meta, flat, extraction_norm = _score_candidate(extraction, dt)
        return {
            "vlm_extraction": extraction_norm,
            "vlm_confidence": score,
        }
    except Exception as e:
        logger.warning("VLM extraction failed: %s", e)
        return {"vlm_extraction": {}, "vlm_confidence": 0.0}
