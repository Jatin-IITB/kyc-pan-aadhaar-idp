from __future__ import annotations

from services.graph.deps import get_deps
from services.graph.state import CaseState


def classify_node(state: CaseState) -> CaseState:
    deps = get_deps()
    dt = (state.get("requested_doc_type") or "auto").lower().strip()
    if dt == "aadhar":
        dt = "aadhaar"

    clf_info = None
    rotation_hint = "rot0"

    if dt in ("auto", ""):
        route = deps.doc_classifier.predict(state["image_bgr"])
        clf_info = {
            "doc_type": route.doc_type,
            "rotation": route.rotation,
            "best_score": route.best_score,
        }
        rotation_hint = route.rotation
        dt = route.doc_type if route.doc_type != "unknown" else "unknown"

    return {
        **state,
        "doc_type": dt,
        "rotation_hint": rotation_hint,
        "classifier_info": clf_info,
    }
