from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.audit.ledger import AuditLedger


class AuditReplayer:
    """Replay audit events to reconstruct case state at any point."""

    _STATE_BUILDERS = {
        "quality_check": lambda s, p: {**s, "quality_passed": p.get("passed", False), "quality_meta": p.get("details", {})},
        "classification": lambda s, p: {**s, "doc_type": p.get("doc_type"), "rotation_hint": p.get("rotation")},
        "extraction": lambda s, p: {**s, "extraction_path": p.get("path"), "yolo_confidence": p.get("yolo_confidence", 0.0), "vlm_confidence": p.get("vlm_confidence", 0.0)},
        "validation": lambda s, p: {**s, "schema_valid": p.get("valid", False), "validation_score": p.get("score", 0.0)},
        "forensics": lambda s, p: {**s, "spoof_score": p.get("spoof_score", 0.0), "forensics_result": p.get("details", {})},
        "policy_check": lambda s, p: {**s, "policy_compliant": p.get("compliant", True)},
        "cross_doc_check": lambda s, p: {**s, "cross_doc_consistency": p.get("consistency_score", 1.0)},
        "decision": lambda s, p: {**s, "decision": p.get("outcome"), "final_confidence": p.get("final_confidence", 0.0)},
        "correction": lambda s, p: {**s, "corrections": s.get("corrections", []) + [{"field": p.get("field_name"), "original": p.get("original_value"), "corrected": p.get("corrected_value"), "reviewer": p.get("reviewer")}]},
        "review": lambda s, p: {**s, "review_outcome": p.get("outcome"), "reviewer": p.get("reviewer"), "review_notes": p.get("notes", "")},
    }

    def replay(self, events: List[Dict[str, Any]], up_to: Optional[int] = None) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        limit = up_to if up_to is not None else len(events)

        for event in events[:limit]:
            event_type = event.get("event_type", "")
            payload = event.get("payload", {})
            builder = self._STATE_BUILDERS.get(event_type)
            if builder:
                state = builder(state, payload)
            state.setdefault("events_replayed", []).append(event_type)

        return state

    def diff(self, events: List[Dict[str, Any]], point_a: int, point_b: int) -> Dict[str, Any]:
        state_a = self.replay(events, up_to=point_a)
        state_b = self.replay(events, up_to=point_b)

        changes = {}
        all_keys = set(state_a.keys()) | set(state_b.keys())
        for key in all_keys:
            if key == "events_replayed":
                continue
            val_a = state_a.get(key)
            val_b = state_b.get(key)
            if val_a != val_b:
                changes[key] = {"before": val_a, "after": val_b}

        return {
            "point_a": point_a,
            "point_b": point_b,
            "changes": changes,
        }
