import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from services.forensics.ela import ELADetector
from services.forensics.font_analysis import FontConsistencyAnalyzer
from services.forensics.metadata import MetadataForensics
from services.forensics.screen_recapture import ScreenRecaptureDetector
from services.forensics.spoof_scorer import SpoofScorer


def _noise(seed=0, h=200, w=300):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_ela_detector_runs():
    result = ELADetector().analyze(_noise())
    assert "ela_score" in result
    assert 0.0 <= result["ela_score"] <= 1.0
    assert "suspicious_regions" in result


def test_screen_recapture_runs():
    result = ScreenRecaptureDetector().detect(_noise())
    assert "is_recaptured" in result
    assert "moire_score" in result


def test_font_analyzer_runs():
    result = FontConsistencyAnalyzer().analyze(_noise(), [])
    assert "font_consistency_score" in result
    assert "inconsistent_regions" in result


def test_spoof_scorer_genuine_all_clean():
    result = SpoofScorer().compute(
        ela_result={"ela_score": 0.001, "suspicious_regions": []},
        copy_move_result={"detected": False, "confidence": 0.0, "matched_pairs": []},
        font_result={"font_consistency_score": 0.9, "inconsistent_regions": []},
        metadata_result={"software_edited": False, "metadata_flags": []},
        screen_result={"is_recaptured": False, "moire_score": 0.05},
    )
    assert result["risk_level"] == "LOW"
    assert result["recommendation"] == "PASS"
    assert result["spoof_score"] < 0.2


def test_spoof_scorer_suspicious_multi_signal():
    result = SpoofScorer().compute(
        ela_result={"ela_score": 0.02, "suspicious_regions": [[0, 0, 20, 20]]},
        copy_move_result={"detected": True, "confidence": 1.0, "matched_pairs": [1] * 20},
        font_result={"font_consistency_score": 0.4, "inconsistent_regions": [[0, 0, 9, 9]]},
        metadata_result={"software_edited": True, "metadata_flags": []},
        screen_result={"is_recaptured": False, "moire_score": 0.1},
    )
    assert result["risk_level"] in ("HIGH", "CRITICAL")
    assert result["spoof_score"] > 0.5


def test_metadata_forensics_handles_empty():
    result = MetadataForensics().analyze(b"")
    assert result["software_edited"] is False


def test_metadata_jpeg_quality_estimation():
    """MetadataForensics estimates JPEG quality from quantization tables."""
    img = _noise(seed=42, h=512, w=768)
    ok, buf_high = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ok, buf_low = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])

    mf = MetadataForensics()
    result_high = mf.analyze(bytes(buf_high))
    result_low = mf.analyze(bytes(buf_low))
    assert result_high["jpeg_quality"] >= 88
    assert result_low["jpeg_quality"] <= 60
    assert "low_jpeg_quality" not in result_high["metadata_flags"]
    assert "low_jpeg_quality" in result_low["metadata_flags"]


def test_metadata_png_has_no_quality():
    """PNG images have no JPEG quantization tables — quality should be None."""
    img = _noise(seed=99, h=200, w=300)
    ok, buf = cv2.imencode(".png", img)
    result = MetadataForensics().analyze(bytes(buf))
    assert result["jpeg_quality"] is None
    assert "low_jpeg_quality" not in result["metadata_flags"]


def test_spoof_scorer_low_jpeg_quality_flags():
    """Low JPEG quality in metadata triggers the metadata gate."""
    result = SpoofScorer().compute(
        ela_result={"ela_score": 0.01, "suspicious_regions": []},
        copy_move_result={"detected": False, "confidence": 0.0, "matched_pairs": []},
        font_result={"font_consistency_score": 0.9, "inconsistent_regions": []},
        metadata_result={"software_edited": False, "metadata_flags": [],
                         "jpeg_quality": 50},
        screen_result={"is_recaptured": False, "moire_score": 0.05},
    )
    assert result["spoof_score"] > 0.2
    assert any(e["type"] == "low_jpeg_quality" for e in result["evidence"])
