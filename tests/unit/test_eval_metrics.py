"""Eval harness metric primitives (Phase 11 W3)."""

from tools.eval.metrics import check_gates, field_match, prf1, score_extraction


def test_field_match_exact_and_fuzzy():
    assert field_match("name", "RAHUL SHARMA", "RAHUL SHARMA") == (True, True)
    # Near-miss surname: fuzzy catches it, exact does not.
    exact, fuzzy = field_match("name", "RAHUL SHARMA", "RAHUL SARMA")
    assert not exact and fuzzy
    # Whitespace and case are normalized before comparison.
    assert field_match("name", "RAHUL  SHARMA", " rahul sharma ")[0] is True


def test_numbers_and_dates_never_match_fuzzily():
    # One wrong digit is a different identity, not a near-miss.
    assert field_match("pan_number", "ABCPE1234F", "ABCPE1235F") == (False, False)
    assert field_match("date_of_birth", "01/02/1990", "01/02/1991") == (False, False)


def test_prf1_math():
    assert prf1(10, 0, 0)["f1"] == 1.0
    assert prf1(0, 5, 5)["f1"] == 0.0
    r = prf1(8, 2, 2)
    assert r["precision"] == 0.8 and r["recall"] == 0.8 and r["f1"] == 0.8


def test_score_extraction_counts_empty_prediction_as_fn():
    samples = [{
        "truth": {"name": "A B", "pan_number": "ABCPE1234F"},
        "predicted": {"name": "A B", "pan_number": ""},
    }]
    r = score_extraction(samples)
    assert r["per_field"]["name"]["f1"] == 1.0
    assert r["per_field"]["pan_number"]["recall"] == 0.0
    assert r["micro"]["recall"] == 0.5


def test_gates_pass_fail_and_skip():
    metrics = {
        "forensics": {"genuine_fpr": 0.0, "overall_recall": 0.44,
                      "per_attack": {"copy_move": {"recall": 1.0}}},
        "decision": {"flagged_leakage": 0, "undetected_autoclear": 41},
        "extraction": None,  # tier did not run -> gates skipped, not failed
    }
    thresholds = {
        "forensics": {"genuine_fpr_max": 0.0, "overall_recall_min": 0.40,
                      "per_attack_recall_min": {"copy_move": 0.95}},
        "decision": {"flagged_leakage_max": 0, "undetected_autoclear_max": 45},
        "extraction": {"micro_f1_min": 0.85},
    }
    g = check_gates(metrics, thresholds)
    assert g["passed"] is True
    statuses = {r["gate"]: r["status"] for r in g["results"]}
    assert statuses["extraction.micro_f1"] == "SKIPPED"

    # A regression must flip the build red.
    metrics["forensics"]["genuine_fpr"] = 0.05
    assert check_gates(metrics, thresholds)["passed"] is False
