from services.cross_doc.contradiction import ContradictionDetector


def test_no_contradictions():
    detector = ContradictionDetector()
    result = detector.detect([
        {"doc_type": "pan", "fields": {"name": "RAHUL SHARMA", "date_of_birth": "15/06/1990"}},
        {"doc_type": "aadhaar", "fields": {"name": "RAHUL SHARMA", "date_of_birth": "15/06/1990"}},
    ])
    assert result["recommendation"] == "PASS"
    assert result["consistency_score"] == 1.0


def test_dob_mismatch_critical():
    detector = ContradictionDetector()
    result = detector.detect([
        {"doc_type": "pan", "fields": {"name": "RAHUL SHARMA", "date_of_birth": "15/06/1990"}},
        {"doc_type": "aadhaar", "fields": {"name": "RAHUL SHARMA", "date_of_birth": "16/06/1990"}},
    ])
    assert result["recommendation"] == "REJECT"
    assert any(c["type"] == "dob_mismatch" for c in result["contradictions"])


def test_gender_mismatch():
    detector = ContradictionDetector()
    result = detector.detect([
        {"doc_type": "pan", "fields": {"name": "RAHUL", "gender": "Male"}},
        {"doc_type": "aadhaar", "fields": {"name": "RAHUL", "gender": "Female"}},
    ])
    assert result["recommendation"] == "REVIEW"
    assert any(c["type"] == "gender_mismatch" for c in result["contradictions"])


def test_single_doc_skipped():
    detector = ContradictionDetector()
    result = detector.detect([
        {"doc_type": "pan", "fields": {"name": "RAHUL"}},
    ])
    assert result["recommendation"] == "PASS"
    assert result["consistency_score"] == 1.0
