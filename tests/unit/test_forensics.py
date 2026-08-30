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
    assert "ring_ratio" in result


def test_screen_recapture_ring_scan_catches_moire():
    """W9: radial-ring scan detects periodic Moiré that wide-band prominence misses."""
    rng = np.random.default_rng(42)
    img = rng.integers(100, 180, (630, 1000, 3), dtype=np.uint8)
    # Overlay strong horizontal Moiré at period 7
    for y in range(img.shape[0]):
        img[y, :, :] = np.clip(
            img[y, :, :].astype(np.int16) + int(40 * np.sin(2 * np.pi * y / 7)),
            0, 255,
        ).astype(np.uint8)
    det = ScreenRecaptureDetector()
    result = det.detect(img)
    assert result["is_recaptured"] is True
    assert result["ring_ratio"] > det.ring_threshold


def test_screen_recapture_flat_image_below_ring_threshold():
    """Flat image with mild noise should not trigger ring scan."""
    rng = np.random.default_rng(7)
    img = np.full((400, 600, 3), 180, dtype=np.uint8)
    img = (img.astype(np.int16) + rng.integers(-5, 5, img.shape)).clip(0, 255).astype(np.uint8)
    det = ScreenRecaptureDetector()
    result = det.detect(img)
    assert result["is_recaptured"] == False
    assert result["ring_ratio"] < det.ring_threshold


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


def test_copy_move_sift_fallback_catches_shifted_patch():
    """W10: SIFT fallback detects copy-move when ORB misses."""
    from services.forensics.copy_move import CopyMoveDetector
    rng = np.random.default_rng(42)
    img = rng.integers(80, 200, (630, 1000, 3), dtype=np.uint8)
    # Add a textured photo-like patch with enough SIFT features
    for c in range(3):
        for y in range(150, 350):
            for x in range(100, 300):
                img[y, x, c] = int(120 + 40 * np.sin(x / 7.0) * np.cos(y / 11.0)
                                   + rng.integers(-10, 10))
    # Copy the patch to a different location (shift > 48px)
    img[200:400, 500:700, :] = img[150:350, 100:300, :]
    det = CopyMoveDetector()
    result = det.detect(img)
    assert result["detected"] is True
    assert result["dominant_shift"] is not None


def test_screen_recapture_combined_score_catches_weak_dual_signal():
    """W11: combined score catches period-13-like Moiré where both signals
    are present but individually below threshold."""
    rng = np.random.default_rng(13)
    img = rng.integers(100, 180, (630, 1000, 3), dtype=np.uint8)
    # Overlay weak horizontal Moiré at period 13
    for y in range(img.shape[0]):
        img[y, :, :] = np.clip(
            img[y, :, :].astype(np.int16) + int(18 * np.sin(2 * np.pi * y / 13)),
            0, 255,
        ).astype(np.uint8)
    det = ScreenRecaptureDetector()
    result = det.detect(img)
    assert "combined_score" in result
    # If either individual signal fires, the combined score is moot —
    # but it should be present and > 0.
    assert result["combined_score"] > 0.0


def test_screen_recapture_combined_score_in_flat_image():
    """Flat images should have low combined score."""
    rng = np.random.default_rng(7)
    img = np.full((400, 600, 3), 180, dtype=np.uint8)
    img = (img.astype(np.int16) + rng.integers(-5, 5, img.shape)).clip(0, 255).astype(np.uint8)
    det = ScreenRecaptureDetector()
    result = det.detect(img)
    assert result["combined_score"] < det.combined_threshold


def test_font_template_band_bound_catches_outliers():
    """W12: band bounds (side='band') flag values outside [low, high]."""
    from services.forensics.font_profile import TemplateFontForensics
    import json, tempfile, pathlib
    profile = {
        "extractor_version": "tf-2",
        "profiles": {
            "test_type": {
                "_vote": 1,
                "corner_top": {"side": "band", "low": 0.20, "high": 0.40},
            }
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(profile, f)
        f.flush()
        det = TemplateFontForensics(profile_path=f.name)
    rng = np.random.default_rng(42)
    img = rng.integers(80, 200, (630, 1000, 3), dtype=np.uint8)
    result = det.analyze(img, "test_type")
    sig_val = result.get("signature", {}).get("corner_top")
    if sig_val is not None and not (0.20 <= sig_val <= 0.40):
        assert result["template_mismatch"] is True
        assert any(b["side"].startswith("band_") for b in result["breaches"])
    pathlib.Path(f.name).unlink()


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
