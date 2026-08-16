"""Noisy-OR aggregation invariants for SpoofScorer (ADR-024)."""

from services.forensics.spoof_scorer import SpoofScorer

CLEAN_ELA = {"ela_score": 0.001, "suspicious_regions": []}
CLEAN_CM = {"detected": False, "confidence": 0.0, "matched_pairs": []}
CLEAN_FONT = {"font_consistency_score": 0.9, "inconsistent_regions": []}
CLEAN_META = {"software_edited": False, "metadata_flags": [], "jpeg_quality": 92}
CLEAN_SCREEN = {"is_recaptured": False, "moire_score": 0.05}


def _score(**over):
    args = dict(ela_result=CLEAN_ELA, copy_move_result=CLEAN_CM, font_result=CLEAN_FONT,
                metadata_result=CLEAN_META, screen_result=CLEAN_SCREEN)
    args.update(over)
    return SpoofScorer().compute(**args)


def test_all_clean_scores_zero_and_passes():
    r = _score()
    assert r["spoof_score"] == 0.0
    assert r["recommendation"] == "PASS"
    assert r["risk_level"] == "LOW"


def test_single_strong_detector_flags_document():
    # The dilution bug: one strong signal must not be averaged into silence.
    r = _score(metadata_result={"software_edited": True, "metadata_flags": []})
    assert r["recommendation"] != "PASS"
    assert r["spoof_score"] >= 0.4


def test_copy_move_detection_drives_reject():
    r = _score(copy_move_result={"detected": True, "confidence": 1.0,
                                 "matched_pairs": [1] * 20, "dominant_shift": [496, 0]})
    assert r["risk_level"] == "CRITICAL"
    assert r["recommendation"] == "REJECT"


def test_font_alone_cannot_flag():
    # Font is corroboration-only: it must never flag a document by itself.
    r = _score(font_result={"font_consistency_score": 0.3, "inconsistent_regions": [[0, 0, 9, 9]]})
    assert r["recommendation"] == "PASS"


def test_font_is_diagnostic_only():
    # Font analysis has no separating power on multi-font ID cards (ADR-026),
    # so it must contribute NOTHING to the score — even a strong font verdict
    # cannot move the spoof score or flag a document on its own.
    base = _score()["spoof_score"]
    withfont = _score(font_result={"font_consistency_score": 0.2,
                                   "inconsistent_regions": [[0, 0, 9, 9]] * 5})
    assert withfont["spoof_score"] == base
    assert withfont["recommendation"] == "PASS"
    assert withfont["component_scores"]["font"] == 0.0


def test_low_jpeg_quality_triggers_review():
    r = _score(metadata_result={"software_edited": False, "metadata_flags": [],
                                "jpeg_quality": 70})
    assert r["recommendation"] != "PASS"
    assert r["spoof_score"] > 0.2
    assert any(e["type"] == "low_jpeg_quality" for e in r["evidence"])


def test_high_jpeg_quality_is_clean():
    r = _score(metadata_result={"software_edited": False, "metadata_flags": [],
                                "jpeg_quality": 92})
    assert r["spoof_score"] == 0.0
    assert r["recommendation"] == "PASS"


def test_noisy_or_is_monotonic_and_bounded():
    r = _score(
        copy_move_result={"detected": True, "confidence": 1.0, "matched_pairs": [1] * 5},
        screen_result={"is_recaptured": True, "moire_score": 0.9},
    )
    assert 0.0 <= r["spoof_score"] <= 1.0
    assert r["spoof_score"] >= 0.9  # at least as strong as the strongest signal
