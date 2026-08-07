from services.decisioning.calibrator import ConfidenceCalibrator
from services.decisioning.auto_clear import AutoClearEngine


def test_perfect_scores_auto_clear():
    cal = ConfidenceCalibrator()
    result = cal.calibrate(
        extraction_score=0.98,
        forensics_score=0.05,
        policy_score=1.0,
        cross_doc_score=1.0,
    )
    assert result["recommendation"] == "AUTO_CLEAR"
    assert result["calibrated_confidence"] > 0.9


def test_low_extraction_review():
    cal = ConfidenceCalibrator()
    result = cal.calibrate(
        extraction_score=0.3,
        forensics_score=0.1,
        policy_score=1.0,
        cross_doc_score=1.0,
    )
    assert result["recommendation"] in ("REVIEW", "REJECT")


def test_high_spoof_reject():
    cal = ConfidenceCalibrator()
    result = cal.calibrate(
        extraction_score=0.95,
        forensics_score=0.8,
        policy_score=1.0,
        cross_doc_score=1.0,
    )
    assert result["calibrated_confidence"] < 0.92


def test_auto_clear_engine_spoof_override():
    cal = ConfidenceCalibrator()
    calibration = cal.calibrate(extraction_score=0.98, forensics_score=0.05)
    engine = AutoClearEngine()
    result = engine.evaluate(calibration, spoof_score=0.6)
    assert result["final_recommendation"] == "REJECT"
    assert len(result["overrides_applied"]) > 0


def test_auto_clear_engine_policy_override():
    cal = ConfidenceCalibrator()
    calibration = cal.calibrate(extraction_score=0.98)
    engine = AutoClearEngine()
    result = engine.evaluate(calibration, policy_compliant=False)
    assert result["final_recommendation"] != "AUTO_CLEAR"


def test_auto_clear_engine_no_overrides():
    cal = ConfidenceCalibrator()
    calibration = cal.calibrate(extraction_score=0.98, forensics_score=0.05)
    engine = AutoClearEngine()
    result = engine.evaluate(calibration, spoof_score=0.1, policy_compliant=True)
    assert result["auto_cleared"] is True
