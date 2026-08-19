"""Tests for template-conformance font forensics (ADR-030)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from services.forensics.font_profile import (
    PROFILE_PATH, TemplateFontForensics, signature,
)
from services.forensics.spoof_scorer import SpoofScorer

REPO_ROOT = Path(__file__).resolve().parents[2]


def _blank(h=400, w=640):
    return np.full((h, w, 3), 255, dtype=np.uint8)


class TestSignature:
    def test_blank_image_has_no_signature(self):
        assert signature(_blank()) is None

    def test_empty_input_is_safe(self):
        assert signature(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    def test_signature_keys_when_text_present(self):
        import cv2
        img = _blank()
        for i in range(6):
            cv2.putText(img, "SAMPLE TEXT 12345", (20, 90 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        sig = signature(img)
        assert sig is not None
        assert set(sig) == {"corner_top", "mod_top", "adv_cv_min"}
        assert sig["corner_top"] > 0


class TestTemplateFontForensics:
    def test_missing_profile_disables_cleanly(self, tmp_path):
        det = TemplateFontForensics(tmp_path / "nope.json")
        assert not det.available
        out = det.analyze(_blank(), "pan")
        assert out["template_mismatch"] is False

    def test_unknown_doc_type_does_not_flag(self):
        det = TemplateFontForensics()
        out = det.analyze(_blank(), "passport")
        assert out["template_mismatch"] is False

    def test_insufficient_text_does_not_flag(self):
        det = TemplateFontForensics()
        if not det.available:
            pytest.skip("no calibrated profile present")
        out = det.analyze(_blank(), "pan")
        assert out["template_mismatch"] is False

    def test_breach_flags_and_reports_feature(self, tmp_path):
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({
            "vote": 1,
            "profiles": {"pan": {"corner_top": {"side": "high", "bound": -1.0}}},
        }))
        det = TemplateFontForensics(prof)

        import cv2
        img = _blank()
        for i in range(6):
            cv2.putText(img, "SAMPLE TEXT 12345", (20, 90 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        out = det.analyze(img, "pan")
        assert out["template_mismatch"] is True
        assert out["breaches"][0]["feature"] == "corner_top"
        assert 0.0 < out["strength"] <= 1.0

    def test_vote_threshold_requires_multiple_breaches(self, tmp_path):
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({
            "vote": 2,
            "profiles": {"pan": {"corner_top": {"side": "high", "bound": -1.0}}},
        }))
        det = TemplateFontForensics(prof)
        import cv2
        img = _blank()
        for i in range(6):
            cv2.putText(img, "SAMPLE TEXT 12345", (20, 90 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        out = det.analyze(img, "pan")
        # one breach, vote of 2 -> must not flag
        assert out["template_mismatch"] is False


class TestScorerIntegration:
    CLEAN = {"jpeg_quality": 92}

    def _score(self, ft):
        return SpoofScorer().compute({}, {}, {}, self.CLEAN, {},
                                     font_template_result=ft)

    def test_absent_result_is_backward_compatible(self):
        out = SpoofScorer().compute({}, {}, {}, self.CLEAN, {})
        assert out["spoof_score"] == 0.0

    def test_no_mismatch_scores_zero(self):
        assert self._score({"template_mismatch": False})["spoof_score"] == 0.0

    def test_mismatch_raises_score(self):
        out = self._score({"template_mismatch": True, "strength": 1.0,
                           "breaches": [{"feature": "corner_top"}]})
        assert out["spoof_score"] > 0.5
        assert any(e["type"] == "font_template_mismatch" for e in out["evidence"])

    def test_legacy_font_prior_still_zero(self):
        # ADR-026 finding stands: intra-document stroke consistency contributes
        # nothing. W6 adds a separate signal rather than reviving this one.
        assert SpoofScorer.DEFAULT_PRIORS["font"] == 0.0
        assert SpoofScorer.DEFAULT_PRIORS["font_template"] > 0.0


class TestCalibratedProfile:
    def test_shipped_profile_records_holdout_measurement(self):
        if not PROFILE_PATH.exists():
            pytest.skip("no calibrated profile committed")
        blob = json.loads(PROFILE_PATH.read_text())
        # The profile must carry its own provenance: a bare threshold file with
        # no recorded out-of-sample result is how unvalidated numbers spread.
        assert "measured_holdout" in blob
        assert blob["measured_holdout"]["fpr"] == 0.0
        assert blob["calibration_seed"] != blob["holdout_seed"]
