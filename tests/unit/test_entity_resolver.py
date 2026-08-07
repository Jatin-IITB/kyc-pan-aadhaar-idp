from services.cross_doc.entity_resolver import EntityResolver, _jaro_winkler, _soundex


def test_jaro_winkler_identical():
    assert _jaro_winkler("RAHUL", "RAHUL") == 1.0


def test_jaro_winkler_similar():
    score = _jaro_winkler("RAHUL KUMAR", "RAHUL KUMR")
    assert score > 0.9


def test_jaro_winkler_different():
    score = _jaro_winkler("RAHUL", "SURESH")
    assert score < 0.7


def test_soundex_basic():
    assert _soundex("RAHUL") == _soundex("RAHOOL")
    assert _soundex("KUMAR") != _soundex("GUPTA")


def test_resolve_names_exact_match():
    resolver = EntityResolver()
    result = resolver.resolve_names([
        {"doc_type": "pan", "value": "RAHUL KUMAR SHARMA"},
        {"doc_type": "aadhaar", "value": "RAHUL KUMAR SHARMA"},
    ])
    assert result["match_score"] == 1.0
    assert result["is_same_person"] is True
    assert len(result["mismatches"]) == 0


def test_resolve_names_minor_variant():
    resolver = EntityResolver()
    result = resolver.resolve_names([
        {"doc_type": "pan", "value": "RAHUL KUMAR SHARMA"},
        {"doc_type": "aadhaar", "value": "RAHUL K SHARMA"},
    ])
    assert result["match_score"] > 0.6
    assert result["is_same_person"] is True


def test_resolve_names_mismatch():
    resolver = EntityResolver()
    result = resolver.resolve_names([
        {"doc_type": "pan", "value": "RAHUL KUMAR SHARMA"},
        {"doc_type": "aadhaar", "value": "SURESH PATEL"},
    ])
    assert result["match_score"] < 0.5
    assert result["is_same_person"] is False


def test_resolve_dates_match():
    resolver = EntityResolver()
    result = resolver.resolve_dates([
        {"doc_type": "pan", "value": "15/06/1990"},
        {"doc_type": "aadhaar", "value": "15/06/1990"},
    ])
    assert result["match"] is True


def test_resolve_dates_mismatch():
    resolver = EntityResolver()
    result = resolver.resolve_dates([
        {"doc_type": "pan", "value": "15/06/1990"},
        {"doc_type": "aadhaar", "value": "16/06/1990"},
    ])
    assert result["match"] is False
    assert len(result["mismatches"]) == 1
