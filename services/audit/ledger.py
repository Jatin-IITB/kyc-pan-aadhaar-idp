from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

GENESIS_HASH = "0" * 64


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_event_hash(prev_hash: str, payload: Any) -> str:
    canonical = _canonical_json(payload)
    data = f"{prev_hash}||{canonical}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class AuditLedger:
    """Hash-chained immutable audit log for KYC case processing."""

    EVENT_TYPES = (
        "quality_check",
        "classification",
        "extraction",
        "validation",
        "forensics",
        "policy_check",
        "cross_doc_check",
        "decision",
        "correction",
        "review",
    )

    def __init__(self) -> None:
        self._chain: List[Dict[str, Any]] = []
        self._head_hash: str = GENESIS_HASH

    def append(self, event_type: str, payload: Dict[str, Any], case_id: Optional[str] = None) -> Dict[str, Any]:
        event_hash = compute_event_hash(self._head_hash, payload)

        event = {
            "case_id": case_id,
            "event_type": event_type,
            "payload": payload,
            "event_hash": event_hash,
            "prev_hash": self._head_hash,
            "sequence": len(self._chain),
        }

        self._chain.append(event)
        self._head_hash = event_hash
        return event

    def verify_chain(self, events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        chain = events if events is not None else self._chain
        if not chain:
            return {"valid": True, "length": 0, "broken_at": None}

        prev_hash = GENESIS_HASH
        for i, event in enumerate(chain):
            expected_prev = event.get("prev_hash", "")
            if expected_prev != prev_hash:
                return {"valid": False, "length": len(chain), "broken_at": i, "reason": "prev_hash mismatch"}

            recomputed = compute_event_hash(prev_hash, event["payload"])
            if recomputed != event.get("event_hash", ""):
                return {"valid": False, "length": len(chain), "broken_at": i, "reason": "event_hash mismatch"}

            prev_hash = event["event_hash"]

        return {"valid": True, "length": len(chain), "broken_at": None}

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self._chain)

    @property
    def head_hash(self) -> str:
        return self._head_hash

    def build_events_from_state(self, state: Dict[str, Any], case_id: str) -> List[Dict[str, Any]]:
        events = []

        if state.get("quality_meta"):
            events.append(self.append("quality_check", {
                "passed": state.get("quality_passed", False),
                "details": state["quality_meta"],
            }, case_id))

        if state.get("doc_type"):
            events.append(self.append("classification", {
                "doc_type": state["doc_type"],
                "rotation": state.get("rotation_hint", "rot0"),
                "classifier_info": state.get("classifier_info"),
            }, case_id))

        if state.get("chosen_extraction"):
            events.append(self.append("extraction", {
                "path": state.get("extraction_path", "yolo"),
                "field_count": len(state.get("chosen_extraction", {})),
                "yolo_confidence": state.get("yolo_confidence", 0.0),
                "vlm_confidence": state.get("vlm_confidence", 0.0),
            }, case_id))

        if state.get("schema_valid") is not None:
            events.append(self.append("validation", {
                "valid": state.get("schema_valid", False),
                "score": state.get("validation_score", 0.0),
                "message": state.get("validation_message", ""),
            }, case_id))

        if state.get("forensics_result"):
            events.append(self.append("forensics", {
                "spoof_score": state.get("spoof_score", 0.0),
                "details": state["forensics_result"],
            }, case_id))

        if state.get("policy_result"):
            events.append(self.append("policy_check", {
                "compliant": state["policy_result"].get("compliant", True),
                "citations": state["policy_result"].get("citations", []),
            }, case_id))

        if state.get("cross_doc_result") and not state["cross_doc_result"].get("skipped"):
            events.append(self.append("cross_doc_check", {
                "consistency_score": state["cross_doc_result"].get("consistency_score", 1.0),
                "contradictions": len(state["cross_doc_result"].get("contradictions", [])),
                "recommendation": state["cross_doc_result"].get("recommendation", "PASS"),
            }, case_id))

        if state.get("decision"):
            events.append(self.append("decision", {
                "outcome": state["decision"],
                "final_confidence": state.get("final_confidence", 0.0),
                "calibration": state.get("calibration_result", {}),
            }, case_id))

        return events
