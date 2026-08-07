from __future__ import annotations

from services.extraction.ensemble import ExtractionEnsemble
from services.graph.nodes.extract_yolo import _score_candidate
from services.graph.state import CaseState


def ensemble_node(state: CaseState) -> CaseState:
    yolo_ext = state.get("yolo_extraction", {})
    yolo_conf = state.get("yolo_confidence", 0.0)
    vlm_ext = state.get("vlm_extraction", {})
    vlm_conf = state.get("vlm_confidence", 0.0)
    dt = state.get("doc_type", "unknown")

    if not vlm_ext:
        return {
            **state,
            "chosen_extraction": yolo_ext,
            "extraction_path": state.get("extraction_path", "yolo"),
        }

    ens = ExtractionEnsemble()
    chosen, path, score = ens.pick_best(yolo_ext, yolo_conf, vlm_ext, vlm_conf, dt)

    if path != state.get("extraction_path", "yolo"):
        score_val, meta, flat, extraction_norm = _score_candidate(chosen, dt)
        return {
            **state,
            "chosen_extraction": extraction_norm,
            "extraction_path": path,
            "extraction_normalized": extraction_norm,
            "flat_fields": flat,
            "schema_valid": meta["is_valid"],
            "validation_message": meta["message"],
            "validation_score": score_val,
        }

    return {
        **state,
        "chosen_extraction": chosen,
        "extraction_path": path,
    }
