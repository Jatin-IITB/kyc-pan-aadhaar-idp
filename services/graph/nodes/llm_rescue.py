from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from services.extraction.normalize import normalize_extraction
from services.graph.deps import get_deps
from services.graph.state import CaseState
from services.pipeline import KYCPipeline
from services.validation.schema_validation import validate_with_schema


def llm_rescue_node(state: CaseState) -> CaseState:
    deps = get_deps()
    if deps.llm_cleaner is None:
        return {"llm_rescue_result": {}}

    if state.get("schema_valid", False):
        return {"llm_rescue_result": {"attempted": False, "reason": "already_valid"}}

    extraction = state.get("extraction_normalized", {})
    if not extraction:
        return {"llm_rescue_result": {}}

    dt = state["doc_type"]
    flat_before = {
        k: (v.get("value") if isinstance(v, dict) else v)
        for k, v in extraction.items()
    }

    try:
        suggestions = deps.llm_cleaner.clean_fields(
            doc_type=dt,
            fields=flat_before,
            failure_reason=state.get("validation_message", ""),
        )
    except Exception as e:
        logger.warning("LLM rescue failed: %s", e)
        return {"llm_rescue_result": {"attempted": True, "error": str(e)}}

    if not isinstance(suggestions, dict) or not suggestions:
        return {"llm_rescue_result": {"attempted": True, "updated": {}}}

    updated = {}
    rejected = {}

    for key, suggested_val in suggestions.items():
        if key not in extraction:
            continue

        field_obj = extraction.get(key)
        if not isinstance(field_obj, dict):
            continue

        orig_val = field_obj.get("value")
        orig_s = KYCPipeline._safe_str(orig_val)
        sug_s = KYCPipeline._safe_str(suggested_val)

        if not sug_s or sug_s == orig_s:
            continue

        accept = False
        if key == "pan_number":
            accept = KYCPipeline._accept_pan_update(orig_s, sug_s)
        elif key == "aadhaar_number":
            accept = KYCPipeline._accept_aadhaar_update(orig_s, sug_s)
        elif key == "date_of_birth":
            accept = KYCPipeline._accept_dob_update(orig_s, sug_s)
        elif key in ("name", "father_name"):
            accept = KYCPipeline._accept_name_update(orig_s, sug_s)

        if not accept:
            rejected[key] = {"original": orig_s, "suggested": sug_s}
            continue

        field_obj["value"] = sug_s
        meta = field_obj.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["source"] = "llm_rescue"
            meta["conf"] = 0.80

        updated[key] = {"original": orig_s, "suggested": sug_s}

    if not updated:
        return {
            "llm_rescue_result": {
                "attempted": True,
                "updated": {},
                "rejected": rejected,
            },
        }

    extraction_norm = normalize_extraction(extraction, dt)
    flat_after = {
        k: (v.get("value") if isinstance(v, dict) else v)
        for k, v in extraction_norm.items()
    }
    is_valid_after, msg_after = validate_with_schema(flat_after, dt)

    return {
        "extraction_normalized": extraction_norm,
        "flat_fields": flat_after,
        "schema_valid": is_valid_after,
        "validation_message": msg_after,
        "llm_rescue_result": {
            "attempted": True,
            "updated": updated,
            "rejected": rejected,
            "after": {"is_valid": is_valid_after, "message": msg_after},
        },
    }
