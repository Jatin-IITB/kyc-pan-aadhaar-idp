from services.cross_doc.entity_resolver import EntityResolver


def test_same_date_different_formats():
    er = EntityResolver()
    result = er.resolve_dates([
        {"value": "15/08/1990", "doc_type": "pan"},
        {"value": "1990-08-15", "doc_type": "aadhaar"},
    ])
    assert result["match"] is True
    assert result["mismatches"] == []


def test_same_date_dd_mm_yyyy_vs_dd_dot_mm_dot_yyyy():
    er = EntityResolver()
    result = er.resolve_dates([
        {"value": "01-12-2000", "doc_type": "pan"},
        {"value": "01.12.2000", "doc_type": "aadhaar"},
    ])
    assert result["match"] is True


def test_different_dates_detected():
    er = EntityResolver()
    result = er.resolve_dates([
        {"value": "15/08/1990", "doc_type": "pan"},
        {"value": "16/08/1990", "doc_type": "aadhaar"},
    ])
    assert result["match"] is False
    assert len(result["mismatches"]) == 1
    assert result["mismatches"][0]["severity"] == "CRITICAL"


def test_unparseable_falls_back_to_string_match():
    er = EntityResolver()
    result = er.resolve_dates([
        {"value": "sometime in 1990", "doc_type": "pan"},
        {"value": "sometime in 1990", "doc_type": "aadhaar"},
    ])
    assert result["match"] is True


def test_unparseable_different_strings():
    er = EntityResolver()
    result = er.resolve_dates([
        {"value": "sometime in 1990", "doc_type": "pan"},
        {"value": "sometime in 1991", "doc_type": "aadhaar"},
    ])
    assert result["match"] is False


def test_named_month_format():
    er = EntityResolver()
    result = er.resolve_dates([
        {"value": "15 Aug 1990", "doc_type": "pan"},
        {"value": "15/08/1990", "doc_type": "aadhaar"},
    ])
    assert result["match"] is True


def test_single_date_always_matches():
    er = EntityResolver()
    result = er.resolve_dates([
        {"value": "01/01/2000", "doc_type": "pan"},
    ])
    assert result["match"] is True
