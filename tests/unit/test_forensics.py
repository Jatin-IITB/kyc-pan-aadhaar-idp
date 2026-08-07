import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from services.forensics.spoof_scorer import SpoofScorer
from services.forensics.ela import ELADetector
from services.forensics.screen_recapture import ScreenRecaptureDetector
from services.forensics.font_analysis import FontConsistencyAnalyzer


def test_spoof_scorer_genuine():
    scorer = SpoofScorer()
    signals = {
        "ela": {"score": 0.1},
        "copy_move": {"score": 0.0},
        "font_consistency": {"score": 0.05},
        "metadata": {"score": 0.1},
        "screen_recapture": {"score": 0.0},
    }
    result = scorer.aggregate(signals)
    assert result["risk_level"] == "LOW"
    assert result["spoof_score"] < 0.2


def test_spoof_scorer_suspicious():
    scorer = SpoofScorer()
    signals = {
        "ela": {"score": 0.8},
        "copy_move": {"score": 0.6},
        "font_consistency": {"score": 0.5},
        "metadata": {"score": 0.7},
        "screen_recapture": {"score": 0.3},
    }
    result = scorer.aggregate(signals)
    assert result["risk_level"] in ("HIGH", "CRITICAL")
    assert result["spoof_score"] > 0.5


def test_ela_detector_runs():
    detector = ELADetector()
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    result = detector.analyze(img)
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0


def test_screen_recapture_runs():
    detector = ScreenRecaptureDetector()
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    result = detector.analyze(img)
    assert "score" in result


def test_font_analyzer_runs():
    analyzer = FontConsistencyAnalyzer()
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    result = analyzer.analyze(img)
    assert "score" in result
