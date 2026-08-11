"""Identity Forge — synthetic Indian ID dataset generator (Phase 11, W1).

Usage:
    python -m tools.forge.identity_forge --type pan --n 200 --out data/synthetic --seed 42
    python -m tools.forge.identity_forge --type all --n 100 --augment full

Per sample, emits:
    <out>/<type>/images/<id>.jpg        rendered document
    <out>/<type>/truth/<id>.json        field values + bboxes + augmentations
    <out>/<type>/labels/<id>.txt        YOLO-format boxes (class per field)
    <out>/<type>/classes.txt            YOLO class list
    <out>/<type>/manifest.jsonl         one line per sample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from faker import Faker

from tools.forge.augment import augment
from tools.forge.identities import FIELD_GENERATORS, make_person
from tools.forge.templates import RENDERERS

DOC_TYPES = list(RENDERERS.keys())


def _yolo_lines(boxes: Dict[str, List[int]], classes: List[str], w: int, h: int) -> List[str]:
    lines = []
    for idx, field in enumerate(classes):
        if field not in boxes:
            continue
        x1, y1, x2, y2 = boxes[field]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def generate(doc_type: str, n: int, out_root: Path, seed: int, augment_level: str) -> List[dict]:
    rng = np.random.default_rng(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    base = out_root / doc_type
    for sub in ("images", "truth", "labels"):
        (base / sub).mkdir(parents=True, exist_ok=True)

    classes = sorted(FIELD_GENERATORS[doc_type](make_person(rng, fake), rng).keys())
    (base / "classes.txt").write_text("\n".join(classes) + "\n")

    render = RENDERERS[doc_type]
    fieldgen = FIELD_GENERATORS[doc_type]
    records = []

    for i in range(n):
        sample_id = f"{doc_type}_{seed}_{i:06d}"
        person = make_person(rng, fake)
        fields = fieldgen(person, rng)

        img, boxes = render(fields, rng)
        img, boxes, applied = augment(img, boxes, augment_level, rng)
        h, w = img.shape[:2]

        cv2.imwrite(str(base / "images" / f"{sample_id}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        truth = {
            "sample_id": sample_id,
            "doc_type": doc_type,
            "fields": fields,
            "boxes": boxes,
            "augmentations": applied,
            "image_size": [w, h],
            "synthetic": True,
        }
        (base / "truth" / f"{sample_id}.json").write_text(json.dumps(truth, indent=2))
        (base / "labels" / f"{sample_id}.txt").write_text(
            "\n".join(_yolo_lines(boxes, classes, w, h)) + "\n")
        records.append(truth)

    with (base / "manifest.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic Indian ID documents")
    ap.add_argument("--type", choices=DOC_TYPES + ["all"], default="pan")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", choices=["none", "light", "full"], default="light")
    args = ap.parse_args()

    types = DOC_TYPES if args.type == "all" else [args.type]
    for t in types:
        records = generate(t, args.n, args.out, args.seed, args.augment)
        print(f"{t}: {len(records)} samples -> {args.out / t}")


if __name__ == "__main__":
    main()
