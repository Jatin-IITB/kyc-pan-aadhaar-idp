from services.audit.replay import AuditReplayer


def _make_event(event_type, payload, seq=0):
    return {"event_type": event_type, "payload": payload, "sequence": seq}


def test_replay_quality_check():
    replayer = AuditReplayer()
    events = [_make_event("quality_check", {"passed": True, "details": {"blur": 0.1}})]
    state = replayer.replay(events)
    assert state["quality_passed"] is True
    assert state["quality_meta"]["blur"] == 0.1


def test_replay_decision():
    replayer = AuditReplayer()
    events = [_make_event("decision", {"outcome": "AUTO_CLEARED", "final_confidence": 0.95})]
    state = replayer.replay(events)
    assert state["decision"] == "AUTO_CLEARED"
    assert state["final_confidence"] == 0.95


def test_replay_correction_event():
    replayer = AuditReplayer()
    events = [
        _make_event("correction", {
            "field_name": "name",
            "original_value": "RAHUL",
            "corrected_value": "RAHUL KUMAR",
            "reviewer": "admin",
        }),
    ]
    state = replayer.replay(events)
    assert len(state["corrections"]) == 1
    assert state["corrections"][0]["field"] == "name"
    assert state["corrections"][0]["corrected"] == "RAHUL KUMAR"


def test_replay_review_event():
    replayer = AuditReplayer()
    events = [
        _make_event("review", {
            "outcome": "APPROVED",
            "reviewer": "admin",
            "notes": "Looks good",
        }),
    ]
    state = replayer.replay(events)
    assert state["review_outcome"] == "APPROVED"
    assert state["reviewer"] == "admin"


def test_replay_up_to_point():
    replayer = AuditReplayer()
    events = [
        _make_event("quality_check", {"passed": True, "details": {}}, seq=0),
        _make_event("classification", {"doc_type": "pan", "rotation": "rot0"}, seq=1),
        _make_event("decision", {"outcome": "SUCCESS", "final_confidence": 0.9}, seq=2),
    ]
    state = replayer.replay(events, up_to=2)
    assert state["doc_type"] == "pan"
    assert "decision" not in state


def test_diff_between_points():
    replayer = AuditReplayer()
    events = [
        _make_event("quality_check", {"passed": True, "details": {}}, seq=0),
        _make_event("classification", {"doc_type": "pan", "rotation": "rot0"}, seq=1),
        _make_event("decision", {"outcome": "SUCCESS", "final_confidence": 0.9}, seq=2),
    ]
    diff = replayer.diff(events, 1, 3)
    assert "doc_type" in diff["changes"]
    assert "decision" in diff["changes"]


def test_unknown_event_type_skipped():
    replayer = AuditReplayer()
    events = [_make_event("unknown_event", {"foo": "bar"})]
    state = replayer.replay(events)
    assert "events_replayed" in state
    assert "unknown_event" in state["events_replayed"]
