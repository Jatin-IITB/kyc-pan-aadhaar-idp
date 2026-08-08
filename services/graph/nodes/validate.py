from __future__ import annotations

from services.extraction.normalize import normalize_extraction
from services.graph.state import CaseState
from services.validation.schema_validation import get_required_fields, validate_with_schema


def validate_node(state: CaseState) -> CaseState:
    extraction = state.get("chosen_extraction") or state.get("yolo_extraction", {})
    dt = state["doc_type"]

    extraction_norm = normalize_extraction(extraction, dt)
    flat = {
        k: (v.get("value") if isinstance(v, dict) else v)
        for k, v in extraction_norm.items()
    }

    required = get_required_fields(dt)
    present_required = sum(
        1 for k in required if k in flat and str(flat[k]).strip() != ""
    )
    coverage = (present_required / max(1, len(required))) if required else 0.0

    confs = []
    for v in extraction_norm.values():
        if isinstance(v, dict):
            confs.append(
                0.5 * float(v.get("det_conf", 0.0))
                + 0.5 * float(v.get("ocr_conf", 0.0))
            )
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

    is_valid, msg = validate_with_schema(flat, dt)
    score = (2.0 if is_valid else 0.0) + coverage + avg_conf

    return {
        "extraction_normalized": extraction_norm,
        "flat_fields": flat,
        "schema_valid": is_valid,
        "validation_message": msg,
        "validation_score": score,
    }
