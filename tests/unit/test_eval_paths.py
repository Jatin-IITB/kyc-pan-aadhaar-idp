"""Regression tests for cross-platform path handling in the eval harness.

The harness maps a sidecar JSON record to its sibling image. The original
implementation used ``str.replace("/attacks/", "/images/")``, which silently
no-ops on Windows because ``glob`` returns backslash-separated paths there --
the sweep then tried to open the .json path with a .jpg suffix and every
forensic run died on the first attack record.
"""
import os
from pathlib import Path

import pytest

from tools.eval.run_eval import _sibling_image


def _p(*parts: str) -> str:
    """Build a path using the running platform's separator."""
    return str(Path(*parts))


def test_attacks_json_maps_to_sibling_image():
    src = _p("data", "tuning", "tamper", "aadhaar", "attacks",
             "aadhaar_42_000000_copy_move_0.json")
    expected = _p("data", "tuning", "tamper", "aadhaar", "images",
                  "aadhaar_42_000000_copy_move_0.jpg")
    assert _sibling_image(src, "attacks") == expected


def test_truth_json_maps_to_sibling_image():
    src = _p("data", "tuning", "synthetic", "pan", "truth", "pan_42_000000.json")
    expected = _p("data", "tuning", "synthetic", "pan", "images", "pan_42_000000.jpg")
    assert _sibling_image(src, "truth") == expected


def test_result_uses_native_separator_and_jpg_suffix():
    out = _sibling_image(
        _p("data", "holdout", "tamper", "pan", "attacks", "x.json"), "attacks")
    assert out.endswith(".jpg")
    assert ".json" not in out
    assert f"{os.sep}images{os.sep}" in out


def test_wrong_sidecar_directory_is_rejected():
    """Fail loudly rather than silently producing an unreadable path."""
    with pytest.raises(ValueError):
        _sibling_image(_p("data", "tuning", "tamper", "pan", "images", "x.json"),
                       "attacks")


def test_stem_containing_dots_is_preserved():
    src = _p("data", "t", "tamper", "pan", "attacks", "pan_1.2_copy_move_0.json")
    assert _sibling_image(src, "attacks").endswith(
        f"{os.sep}images{os.sep}pan_1.2_copy_move_0.jpg")
