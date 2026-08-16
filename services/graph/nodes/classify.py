from __future__ import annotations

import logging

from services.graph.deps import get_deps
from services.graph.state import CaseState

logger = logging.getLogger(__name__)


def classify_node(state: CaseState) -> CaseState:
    deps = get_deps()
    dt = (state.get("requested_doc_type") or "auto").lower().strip()
    if dt == "aadhar":
        dt = "aadhaar"

    clf_info = None
    rotation_hint = "rot0"
    model_rotation = None

    # Only consult the model when it is confident. A wrong rotation is unrecoverable
    # downstream (extraction reads a sideways card), so the detector-based search
    # stays authoritative unless the model clearly beats the uncertainty bar.
    if deps.rotation_classifier is not None:
        rot_label, rot_conf = deps.rotation_classifier.predict(state["image_bgr"])
        min_conf = getattr(deps.config, "rotation_min_confidence", 0.90)
        if rot_conf >= min_conf:
            model_rotation = rot_label
            rotation_hint = rot_label
        logger.info(
            "Rotation classifier: %s (conf=%.3f, min=%.2f, used=%s)",
            rot_label, rot_conf, min_conf, model_rotation is not None,
        )

    if dt in ("auto", ""):
        route = deps.doc_classifier.predict(state["image_bgr"])
        clf_info = {
            "doc_type": route.doc_type,
            "rotation": route.rotation,
            "best_score": route.best_score,
        }
        if model_rotation is None:
            rotation_hint = route.rotation
        dt = route.doc_type if route.doc_type != "unknown" else "unknown"

    return {
        "doc_type": dt,
        "rotation_hint": rotation_hint,
        "classifier_info": clf_info,
    }
