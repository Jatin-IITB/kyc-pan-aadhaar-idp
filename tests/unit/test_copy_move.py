import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from services.forensics.copy_move import CopyMoveDetector


def _noise_doc(seed: int = 7, h: int = 512, w: int = 768) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_pristine_noise_is_clean():
    res = CopyMoveDetector().detect(_noise_doc())
    assert res["detected"] is False


def test_planted_copy_move_is_detected():
    img = _noise_doc()
    # Duplicate a 128x128 region, shifted by a grid-aligned offset.
    img[192:320, 448:576] = img[192:320, 64:192]
    res = CopyMoveDetector().detect(img)
    assert res["detected"] is True
    assert res["confidence"] >= 0.5
    # All evidence pairs must agree on the planted displacement (384, 0).
    assert res["dominant_shift"] == [384, 0]


def test_periodic_print_pattern_is_clean():
    # Guilloche/watermark analog: a tiled texture is pixel-identical everywhere.
    # This is the genuine-PAN failure mode that motivated ADR-022.
    rng = np.random.default_rng(7)
    tile = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    img = np.tile(tile, (16, 24, 1))
    res = CopyMoveDetector().detect(img)
    assert res["detected"] is False


def test_flat_image_is_clean():
    img = np.full((512, 768, 3), 200, dtype=np.uint8)
    res = CopyMoveDetector().detect(img)
    assert res["detected"] is False
    assert res["confidence"] == 0.0
