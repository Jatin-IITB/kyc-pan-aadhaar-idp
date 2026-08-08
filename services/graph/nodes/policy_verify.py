from __future__ import annotations

import logging

from services.graph.deps import get_deps
from services.graph.state import CaseState

logger = logging.getLogger(__name__)


def policy_verify_node(state: CaseState) -> CaseState:
    deps = get_deps()
    policy_verifier = getattr(deps, "policy_verifier", None)

    if policy_verifier is None:
        return {"policy_result": {}}

    flat_fields = state.get("flat_fields", {})
    if not flat_fields:
        return {"policy_result": {}}

    case_data = {
        "doc_type": state.get("doc_type", "unknown"),
        "flat_fields": flat_fields,
    }

    try:
        result = policy_verifier.verify(case_data)
    except Exception:
        logger.exception("Policy verification failed")
        result = {"overall_status": "REQUIRES_REVIEW", "checks": [], "error": "verification_failed"}

    return {"policy_result": result}
