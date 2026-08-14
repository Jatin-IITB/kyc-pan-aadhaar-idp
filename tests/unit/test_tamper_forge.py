import json

import numpy as np
import pytest

pytest.importorskip("faker")
cv2 = pytest.importorskip("cv2")

from tools.forge.identity_forge import generate
from tools.forge.tamper_forge import ATTACKS, forge_dataset


@pytest.fixture(scope="module")
def genuine(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    generate("pan", 2, root, seed=5, augment_level="none")
    return root


def test_all_attacks_emit_labeled_output(genuine, tmp_path):
    records = forge_dataset(genuine, tmp_path, ["pan"], ATTACKS,
                            per_doc=1, severity="med", seed=9)
    assert {r["attack"] for r in records} == set(ATTACKS)
    for r in records:
        img = cv2.imread(str(tmp_path / "pan" / "images" / f"{r['sample_id']}.jpg"))
        assert img is not None
        assert r["targets"]  # each attack declares which detectors it should trip
        label = json.loads((tmp_path / "pan" / "attacks" / f"{r['sample_id']}.json").read_text())
        assert label["attack"] == r["attack"]


def test_copy_move_attack_is_detected(genuine, tmp_path):
    from services.forensics.copy_move import CopyMoveDetector

    forge_dataset(genuine, tmp_path, ["pan"], ["copy_move"], per_doc=1, severity="high", seed=1)
    imgs = list((tmp_path / "pan" / "images").glob("*copy_move*.jpg"))
    assert imgs
    detected = [CopyMoveDetector().detect(cv2.imread(str(p)))["detected"] for p in imgs]
    assert any(detected)


def test_exif_attack_sets_software_tag(genuine, tmp_path):
    from services.forensics.metadata import MetadataForensics

    forge_dataset(genuine, tmp_path, ["pan"], ["exif_edit"], per_doc=1, severity="high", seed=1)
    p = next((tmp_path / "pan" / "images").glob("*exif_edit*.jpg"))
    meta = MetadataForensics().analyze(p.read_bytes())
    assert meta["software_edited"] is True


def test_copy_move_offset_is_not_grid_aligned(genuine, tmp_path):
    """Audit S4: verify the forge generates non-grid-aligned offsets, so the
    eval isn't trivially passing because attacks happen to align to the
    detector's sampling stride."""
    forge_dataset(genuine, tmp_path, ["pan"], ["copy_move"], per_doc=1, severity="high", seed=42)
    labels = list((tmp_path / "pan" / "attacks").glob("*copy_move*.json"))
    assert labels
    for lp in labels:
        rec = json.loads(lp.read_text())
        dx, dy = rec["params"]["offset"]
        # At least one axis must NOT be a multiple of 16 (old detector stride).
        assert dx % 16 != 0 or dy % 16 != 0, f"offset {dx},{dy} is grid-aligned"


def test_screen_evasion_case_not_detected(genuine, tmp_path):
    """Audit M7: the low-severity screen attack (3px grid) is a deliberate
    evasion case — the detector should NOT fire on it."""
    from services.forensics.screen_recapture import ScreenRecaptureDetector

    forge_dataset(genuine, tmp_path, ["pan"], ["screen_recapture"],
                  per_doc=1, severity="low", seed=7)
    imgs = list((tmp_path / "pan" / "images").glob("*screen_recapture*.jpg"))
    assert imgs
    det = ScreenRecaptureDetector()
    for p in imgs:
        res = det.detect(cv2.imread(str(p)))
        assert not res["is_recaptured"], "3px evasion case should not be detected"
