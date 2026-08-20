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

    def test_grayscale_input_does_not_crash(self):
        # Audit S1: 2-D input formerly raised cv2.error, which the forensics
        # node's blanket except converted into a silent PASS (fail-open).
        assert signature(np.full((400, 640), 255, dtype=np.uint8)) is None

    def test_bgra_input_does_not_crash(self):
        assert signature(np.full((400, 640, 4), 255, dtype=np.uint8)) is None

    def test_signature_keys_when_text_present(self):
        import cv2
        img = _blank()
        for i in range(6):
            cv2.putText(img, "SAMPLE TEXT 12345", (20, 90 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        sig = signature(img)
        assert sig is not None
        assert set(sig) == {"corner_top", "mod_top", "adv_cv_min", "id_width_cv"}
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

    def test_malformed_spec_is_skipped_not_crashed(self, tmp_path):
        # Audit S1: a config typo (missing "bound") formerly raised KeyError,
        # zeroing the entire forensics result via the node's blanket except.
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({
            "vote": 1,
            "profiles": {"pan": {"corner_top": {"side": "high"},          # no bound
                                 "mod_top": {"bound": 0.1},               # no side
                                 "adv_cv_min": "not-a-dict"}},
        }))
        det = TemplateFontForensics(prof)
        import cv2
        img = _blank()
        for i in range(6):
            cv2.putText(img, "SAMPLE TEXT 12345", (20, 90 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        out = det.analyze(img, "pan")
        assert out["template_mismatch"] is False

    def test_low_side_value_above_bound_does_not_fire(self, tmp_path):
        # Direction semantics: a "low"-side envelope flags values BELOW the
        # bound; a value above it is inside the genuine envelope.
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({
            "vote": 1,
            "profiles": {"pan": {"mod_top": {"side": "low", "bound": -100.0}}},
        }))
        det = TemplateFontForensics(prof)
        import cv2
        img = _blank()
        for i in range(6):
            cv2.putText(img, "SAMPLE TEXT 12345", (20, 90 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        out = det.analyze(img, "pan")
        assert out["template_mismatch"] is False

    def test_extractor_version_mismatch_refuses_profile(self, tmp_path):
        # Audit S3: envelopes are only valid for the extractor that produced
        # them; a version mismatch must disable, not silently mis-score.
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({
            "extractor_version": "tf-0-ancient",
            "vote": 1,
            "profiles": {"pan": {"corner_top": {"side": "high", "bound": -1.0}}},
        }))
        det = TemplateFontForensics(prof)
        assert not det.available

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

    def test_per_doc_type_vote_overrides_global(self, tmp_path):
        """W8: _vote in per-doc-type profile overrides the global vote."""
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({
            "vote": 1,
            "profiles": {
                "pan": {
                    "_vote": 2,
                    "corner_top": {"side": "high", "bound": -1.0},
                },
            },
        }))
        det = TemplateFontForensics(prof)
        import cv2
        img = _blank()
        for i in range(6):
            cv2.putText(img, "SAMPLE TEXT 12345", (20, 90 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        out = det.analyze(img, "pan")
        # global vote=1 would flag, but per-type _vote=2 requires 2 breaches
        assert out["template_mismatch"] is False

    def test_vote_metadata_key_not_treated_as_feature(self, tmp_path):
        """W8: _vote must not be iterated as a feature spec."""
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({
            "vote": 1,
            "profiles": {
                "pan": {
                    "_vote": 1,
                    "corner_top": {"side": "high", "bound": -1.0},
                },
            },
        }))
        det = TemplateFontForensics(prof)
        import cv2
        img = _blank()
        for i in range(6):
            cv2.putText(img, "SAMPLE TEXT 12345", (20, 90 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        out = det.analyze(img, "pan")
        assert out["template_mismatch"] is True
        # _vote should never appear in breach list
        assert all(b["feature"] != "_vote" for b in out["breaches"])


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
