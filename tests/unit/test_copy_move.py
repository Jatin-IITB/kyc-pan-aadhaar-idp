import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from services.forensics.copy_move import CopyMoveDetector


def _textured(seed: int = 7, h: int = 512, w: int = 768) -> np.ndarray:
    """Rich-texture image: ORB needs corners, like a real photographed doc."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_pristine_is_clean():
    assert CopyMoveDetector().detect(_textured())["detected"] is False


def test_grid_aligned_duplication_detected():
    img = _textured()
    img[192:320, 448:576] = img[192:320, 64:192]  # offset (384, 0)
    res = CopyMoveDetector().detect(img)
    assert res["detected"] is True
    assert res["dominant_shift"] == [384, 0]


def test_non_grid_aligned_duplication_detected():
    # The audit's proven-blind case for the v2 grid detector: an offset that is
    # NOT a multiple of any sampling stride. ORB is anchored to content, so it
    # must still fire (ADR-026).
    img = _textured()
    img[100:228, 60:188] = img[103:231, 450:578]  # offset (-390, -3)
    res = CopyMoveDetector().detect(img)
    assert res["detected"] is True


def test_periodic_pattern_is_clean():
    # Tiled texture is pixel-identical everywhere — the genuine-PAN guilloche
    # failure mode. Same-offset matches scatter card-wide and fail dominance.
    rng = np.random.default_rng(7)
    tile = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert CopyMoveDetector().detect(np.tile(tile, (16, 24, 1)))["detected"] is False


def test_thin_repeated_strip_is_clean():
    # A duplicated single text line is a wide, short strip (high aspect ratio) —
    # out of scope for region copy-move, must not false-positive.
    img = _textured()
    strip = img[240:260, 100:400].copy()
    img[240:260, 420:720] = strip
    assert CopyMoveDetector().detect(img)["detected"] is False


def test_flat_image_is_clean():
    img = np.full((512, 768, 3), 200, dtype=np.uint8)
    res = CopyMoveDetector().detect(img)
    assert res["detected"] is False and res["confidence"] == 0.0
