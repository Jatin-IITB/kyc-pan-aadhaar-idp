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
