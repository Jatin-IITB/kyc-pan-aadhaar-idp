from __future__ import annotations

import logging

from services.audit.ledger import AuditLedger
from services.graph.state import CaseState

logger = logging.getLogger(__name__)


def audit_commit_node(state: CaseState) -> CaseState:
    case_id = state.get("case_id", "unknown")
    ledger = AuditLedger()
    events = ledger.build_events_from_state(state, case_id)

    logger.info("audit_commit: %d events, head=%s", len(events), ledger.head_hash[:12])

    return {**state, "audit_events": events}
