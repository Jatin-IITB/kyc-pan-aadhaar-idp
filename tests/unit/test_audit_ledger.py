from services.audit.ledger import AuditLedger, GENESIS_HASH, compute_event_hash


def test_genesis_hash():
    assert len(GENESIS_HASH) == 64
    assert GENESIS_HASH == "0" * 64


def test_append_and_verify():
    ledger = AuditLedger()
    ledger.append("quality_check", {"passed": True}, case_id="case-1")
    ledger.append("classification", {"doc_type": "pan"}, case_id="case-1")
    ledger.append("decision", {"outcome": "SUCCESS"}, case_id="case-1")

    result = ledger.verify_chain()
    assert result["valid"] is True
    assert result["length"] == 3


def test_tamper_detection():
    ledger = AuditLedger()
    ledger.append("quality_check", {"passed": True}, case_id="case-1")
    ledger.append("decision", {"outcome": "SUCCESS"}, case_id="case-1")

    events = ledger.get_events()
    events[0]["payload"]["passed"] = False

    result = ledger.verify_chain(events)
    assert result["valid"] is False
    assert result["broken_at"] == 0


def test_hash_deterministic():
    h1 = compute_event_hash("abc", {"key": "value"})
    h2 = compute_event_hash("abc", {"key": "value"})
    assert h1 == h2


def test_build_events_from_state():
    ledger = AuditLedger()
    state = {
        "quality_meta": {"blur": False},
        "quality_passed": True,
        "doc_type": "pan",
        "rotation_hint": "rot0",
        "chosen_extraction": {"pan_number": {"value": "ABCDE1234F"}},
        "extraction_path": "yolo",
        "schema_valid": True,
        "validation_score": 0.95,
        "validation_message": "valid",
        "decision": "AUTO_CLEARED",
        "final_confidence": 0.95,
    }
    events = ledger.build_events_from_state(state, "case-1")
    assert len(events) >= 4

    verification = ledger.verify_chain()
    assert verification["valid"] is True
