from services.decisioning.thresholds import (
    SPOOF_AUTO_CLEAR_OVERRIDE,
    SPOOF_REJECT_THRESHOLD,
    SPOOF_REVIEW_THRESHOLD,
)


def test_threshold_ordering():
    assert SPOOF_REVIEW_THRESHOLD < SPOOF_AUTO_CLEAR_OVERRIDE < SPOOF_REJECT_THRESHOLD


def test_thresholds_in_unit_range():
    for t in (SPOOF_REJECT_THRESHOLD, SPOOF_AUTO_CLEAR_OVERRIDE, SPOOF_REVIEW_THRESHOLD):
        assert 0.0 <= t <= 1.0
