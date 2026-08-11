import json
import re

import numpy as np
import pytest

pytest.importorskip("faker")
cv2 = pytest.importorskip("cv2")

from tools.forge.augment import augment
from tools.forge.identities import (
    aadhaar_number,
    pan_number,
    verhoeff_check_digit,
    verhoeff_validate,
)
from tools.forge.identity_forge import generate


def test_verhoeff_known_vector():
    # Classic reference vector: check digit of "236" is 3.
    assert verhoeff_check_digit("236") == "3"
    assert verhoeff_validate("2363")
    assert not verhoeff_validate("2364")


def test_aadhaar_numbers_are_verhoeff_valid():
    rng = np.random.default_rng(0)
    for _ in range(50):
        num = aadhaar_number(rng).replace(" ", "")
        assert len(num) == 12
        assert num[0] in "23456789"
        assert verhoeff_validate(num)


def test_pan_grammar_and_surname_link():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = pan_number(rng, "SHARMA")
        assert re.fullmatch(r"[A-Z]{3}P S?[A-Z][0-9]{4}[A-Z]".replace(" ", ""), p)
        assert p[3] == "P" and p[4] == "S"


def test_generate_emits_consistent_artifacts(tmp_path):
    records = generate("pan", 3, tmp_path, seed=7, augment_level="light")
    assert len(records) == 3
    for r in records:
        img = cv2.imread(str(tmp_path / "pan" / "images" / f"{r['sample_id']}.jpg"))
        assert img is not None
        h, w = img.shape[:2]
        truth = json.loads((tmp_path / "pan" / "truth" / f"{r['sample_id']}.json").read_text())
        assert truth["fields"].keys() == {"pan_number", "name", "father_name", "date_of_birth"}
        for x1, y1, x2, y2 in truth["boxes"].values():
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        labels = (tmp_path / "pan" / "labels" / f"{r['sample_id']}.txt").read_text().split()
        assert all(0 <= float(v) <= 1 for v in labels if "." in v)


def test_rot90_transforms_boxes_correctly():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (200, 400, 3), dtype=np.uint8)
    boxes = {"f": [40, 20, 120, 60]}
    marker = img[20:60, 40:120].copy()

    out, new_boxes, applied = img, boxes, []
    # Force a deterministic 90cw rotation via the internal helper.
    from tools.forge.augment import _rot90
    out, new_boxes = _rot90(img, boxes, 1)
    x1, y1, x2, y2 = new_boxes["f"]
    assert (out[y1:y2, x1:x2].shape[:2]) == (marker.shape[1], marker.shape[0])
    # Content inside the transformed box is the rotated marker.
    assert np.array_equal(out[y1:y2, x1:x2], cv2.rotate(marker, cv2.ROTATE_90_CLOCKWISE))


def test_augment_none_is_identity():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    boxes = {"f": [10, 10, 50, 50]}
    out, out_boxes, applied = augment(img, boxes, "none", rng)
    assert applied == []
    assert np.array_equal(out, img) and out_boxes == boxes
