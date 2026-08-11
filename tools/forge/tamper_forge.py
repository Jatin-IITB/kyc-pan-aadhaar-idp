"""Tamper Forge — programmatic forgery generator (Phase 11, W2).

Red-teams our own forensics suite. Takes genuine synthetic documents (from the
Identity Forge) and emits labeled forged variants across six attack classes,
each parameterized by severity so we can plot recall-vs-subtlety curves.

    python -m tools.forge.tamper_forge --in data/synthetic --out data/tamper \
        --attacks all --per-doc 1 --severity mixed

Each forged sample carries an ``attack.json`` recording the class, target
region, severity, and which detector(s) it is designed to trip.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tools.forge.fonts import font_override
from tools.forge.identities import FIELD_GENERATORS, make_person
from tools.forge.templates import RENDERERS

from faker import Faker

Region = List[int]  # [x1, y1, x2, y2]


@dataclass
class AttackLabel:
    attack: str
    severity: str
    region: Optional[Region]
    targets: List[str]  # detectors expected to fire
    params: Dict[str, object] = field(default_factory=dict)


# --- individual attacks ---------------------------------------------------
# Each returns (forged_image, AttackLabel). Genuine truth JSON is unchanged;
# the forgery is a pixel/EXIF manipulation, not a field-value edit, except
# text_splice which also perturbs a rendered value.


def _rand_field_box(truth: dict, rng: np.random.Generator) -> Tuple[str, Region]:
    fields = [k for k in truth["boxes"] if not k.startswith("_")]
    field_name = str(rng.choice(fields))
    return field_name, truth["boxes"][field_name]


def attack_copy_move(img, truth, rng, severity) -> Tuple[np.ndarray, AttackLabel]:
    """Duplicate the photo region to an ARBITRARY location.

    The canonical ID copy-move hides or replaces content with the holder's
    photo/stamp — 2-D textured content. The destination offset is drawn with
    single-pixel jitter so it is never aligned to any detector sampling grid
    (the v2 detector's blindness to non-aligned offsets was audit finding C2).
    """
    h, w = img.shape[:2]
    box = truth["boxes"].get("_photo")
    if box is None:  # fallback: largest field box
        box = max((b for k, b in truth["boxes"].items() if not k.startswith("_")),
                  key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    # Duplicate the whole photo region (the realistic "clone the face" forgery);
    # severity trims how much of it, but never below a detectable region size.
    bw, bh = box[2] - box[0], box[3] - box[1]
    frac = {"low": 0.75, "med": 0.9, "high": 1.0}[severity]
    rw, rh = int(bw * frac), int(bh * frac)
    rw, rh = min(rw, w // 2), min(rh, h // 2)
    sx, sy = box[0], box[1]

    # Arbitrary non-overlapping destination with sub-block jitter, so the
    # offset is never aligned to any detector sampling grid (audit C2).
    for _ in range(50):
        dx = int(rng.integers(0, w - rw))
        dy = int(rng.integers(0, h - rh))
        if abs(dx - sx) > rw or abs(dy - sy) > rh:
            break
    out = img.copy()
    out[dy:dy + rh, dx:dx + rw] = img[sy:sy + rh, sx:sx + rw]
    return out, AttackLabel(
        "copy_move", severity, [dx, dy, dx + rw, dy + rh],
        ["copy_move"], {"src": [sx, sy], "size": [rw, rh],
                        "offset": [dx - sx, dy - sy]},
    )


def attack_text_splice(img, truth, rng, severity) -> Tuple[np.ndarray, AttackLabel]:
    """Paint over one field and stamp a different value in a mismatched font."""
    from PIL import Image, ImageDraw
    from tools.forge.fonts import load_font

    field_name, box = _rand_field_box(truth, rng)
    x1, y1, x2, y2 = box
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    # Cover the original value with a sampled background swatch (leaves a seam).
    swatch = img[max(0, y1 - 6):y1 - 1, x1:x2].mean(axis=(0, 1)) if y1 > 6 else np.array([245, 245, 240])
    d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], fill=tuple(int(c) for c in swatch[::-1]))
    # New value: perturb digits / letters of the original.
    orig = truth["fields"].get(field_name, "TAMPERED")
    tampered = _perturb_value(orig, rng)
    size = max(18, y2 - y1)
    wrong_font = {"low": "regular", "med": "serif", "high": "mono"}[severity]
    d.text((x1, y1), tampered, font=load_font(wrong_font, size), fill=(15, 15, 20))
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Realistic splice carries a different compression history: the edited
    # patch is JPEG-cycled harder than the host, leaving an ELA-visible seam.
    patch_q = {"low": 75, "med": 60, "high": 45}[severity]
    py1, py2 = max(0, y1 - 6), min(out.shape[0], y2 + 6)
    px1, px2 = max(0, x1 - 6), min(out.shape[1], x2 + 6)
    patch = out[py1:py2, px1:px2]
    ok, buf = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, patch_q])
    out[py1:py2, px1:px2] = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return out, AttackLabel(
        "text_splice", severity, [x1, y1, x2, y2],
        ["ela", "font"], {"field": field_name, "from": orig, "to": tampered},
    )


def attack_font_swap(img, truth, rng, severity, doc_type) -> Tuple[np.ndarray, AttackLabel]:
    """Re-render the SAME card (via its recorded render seed) with mismatched
    value typography — the only delta from the genuine document is the fonts."""
    swaps = {"low": {"bold": "serif"}, "med": {"bold": "serif", "mono": "regular"},
             "high": {"bold": "mono", "mono": "serif", "regular": "serif"}}[severity]
    render_rng = np.random.default_rng(truth.get("render_seed", 0))
    with font_override(swaps):
        out, _ = RENDERERS[doc_type](truth["fields"], render_rng)
    return out, AttackLabel("font_swap", severity, None, ["font"], {"swaps": swaps})


def attack_screen_recapture(img, truth, rng, severity) -> Tuple[np.ndarray, AttackLabel]:
    """Simulate a photo-of-screen: Moire beat + glare.

    Severity maps to crudeness — and "low" is a deliberate EVASION case
    (audit C2): a 3 px grid sits above the FFT detector's mid-band cutoff, so
    the eval's recall curve includes attacks the detector cannot see, instead
    of only generating detector-visible periods.
    """
    h, w = img.shape[:2]
    period = {"low": 3, "med": 13, "high": 7}[severity]
    amp = {"low": 12, "med": 18, "high": 26}[severity]
    yy, xx = np.mgrid[0:h, 0:w]
    # Anisotropic beat: strong vertical scanline + weaker horizontal.
    moire = (np.sin(2 * np.pi * xx / period)
             + 0.6 * np.sin(2 * np.pi * yy / (period + 2)))
    out = np.clip(img.astype(np.float32) + amp * moire[..., None], 0, 255)
    # Specular glare blob.
    gx, gy = int(rng.uniform(0.3, 0.7) * w), int(rng.uniform(0.2, 0.5) * h)
    glare = np.zeros((h, w), np.float32)
    cv2.circle(glare, (gx, gy), int(0.18 * w), 1.0, -1)
    glare = cv2.GaussianBlur(glare, (0, 0), 0.12 * w)
    out = np.clip(out + (40 * glare)[..., None], 0, 255).astype(np.uint8)
    return out, AttackLabel("screen_recapture", severity, None, ["screen"],
                            {"period": period, "amp": amp})


def attack_exif_edit(img, truth, rng, severity) -> Tuple[np.ndarray, AttackLabel, bytes]:
    """Re-encode with editor software tag (+ date mismatch at higher severity)."""
    from io import BytesIO

    from PIL import Image

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    exif = pil.getexif()
    editor = {"low": "GIMP 2.10", "med": "Adobe Photoshop 24.0",
              "high": "Adobe Photoshop CC 2023"}[severity]
    exif[0x0131] = editor  # Software
    if severity in ("med", "high"):
        exif[0x0132] = "2023:11:02 10:15:00"  # DateTime (modified)
        exif[0x9003] = "2019:03:14 08:00:00"  # DateTimeOriginal
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=92, exif=exif)
    data = buf.getvalue()
    out = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return out, AttackLabel("exif_edit", severity, None, ["metadata"],
                            {"software": editor}), data


def attack_regenerate(img, truth, rng, severity, doc_type) -> Tuple[np.ndarray, AttackLabel]:
    """Re-render the same identity, then double-JPEG at low quality (ELA seam)."""
    out, _ = RENDERERS[doc_type](truth["fields"], np.random.default_rng(999))
    q = {"low": 75, "med": 60, "high": 45}[severity]
    for _ in range(2):
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out, AttackLabel("regenerate", severity, None, ["ela", "font"], {"quality": q})


# --- helpers --------------------------------------------------------------

def _perturb_value(value: str, rng: np.random.Generator) -> str:
    chars = list(value)
    digits = [i for i, c in enumerate(chars) if c.isdigit()]
    letters = [i for i, c in enumerate(chars) if c.isalpha()]
    pool = digits or letters
    if not pool:
        return value + "X"
    for i in rng.choice(pool, size=min(2, len(pool)), replace=False):
        chars[i] = str(rng.integers(0, 10)) if chars[i].isdigit() else chr(int(rng.integers(65, 91)))
    return "".join(chars)


ATTACKS = ["copy_move", "text_splice", "font_swap", "screen_recapture", "exif_edit", "regenerate"]
_SEV_LEVELS = ["low", "med", "high"]


def _apply(name, img, truth, rng, severity, doc_type):
    if name == "copy_move":
        return attack_copy_move(img, truth, rng, severity)
    if name == "text_splice":
        return attack_text_splice(img, truth, rng, severity)
    if name == "font_swap":
        return attack_font_swap(img, truth, rng, severity, doc_type)
    if name == "screen_recapture":
        return attack_screen_recapture(img, truth, rng, severity)
    if name == "regenerate":
        return attack_regenerate(img, truth, rng, severity, doc_type)
    if name == "exif_edit":
        return attack_exif_edit(img, truth, rng, severity)
    raise ValueError(name)


def forge_dataset(in_root: Path, out_root: Path, doc_types: List[str],
                  attacks: List[str], per_doc: int, severity: str, seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    records: List[dict] = []

    for doc_type in doc_types:
        src = in_root / doc_type
        truths = sorted((src / "truth").glob("*.json")) if (src / "truth").exists() else []
        if not truths:
            continue
        base = out_root / doc_type
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "attacks").mkdir(parents=True, exist_ok=True)

        for tp in truths:
            truth = json.loads(tp.read_text())
            img = cv2.imread(str(src / "images" / f"{truth['sample_id']}.jpg"))
            if img is None:
                continue
            for attack in attacks:
                for k in range(per_doc):
                    sev = severity if severity != "mixed" else _SEV_LEVELS[rng.integers(0, 3)]
                    result = _apply(attack, img, truth, rng, sev, doc_type)
                    exif_bytes = None
                    if len(result) == 3:
                        forged, label, exif_bytes = result
                    else:
                        forged, label = result

                    fid = f"{truth['sample_id']}_{attack}_{k}"
                    out_path = base / "images" / f"{fid}.jpg"
                    if exif_bytes is not None:
                        out_path.write_bytes(exif_bytes)
                    else:
                        cv2.imwrite(str(out_path), forged, [cv2.IMWRITE_JPEG_QUALITY, 92])

                    rec = {
                        "sample_id": fid,
                        "source_id": truth["sample_id"],
                        "doc_type": doc_type,
                        "fields": truth["fields"],
                        **asdict(label),
                    }
                    (base / "attacks" / f"{fid}.json").write_text(json.dumps(rec, indent=2))
                    records.append(rec)

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "manifest.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate labeled forged documents")
    ap.add_argument("--in", dest="in_root", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--out", type=Path, default=Path("data/tamper"))
    ap.add_argument("--types", nargs="+", default=list(RENDERERS.keys()))
    ap.add_argument("--attacks", nargs="+", default=["all"])
    ap.add_argument("--per-doc", type=int, default=1)
    ap.add_argument("--severity", choices=_SEV_LEVELS + ["mixed"], default="mixed")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    attacks = ATTACKS if args.attacks == ["all"] else args.attacks
    records = forge_dataset(args.in_root, args.out, args.types, attacks,
                            args.per_doc, args.severity, args.seed)
    by_attack: Dict[str, int] = {}
    for r in records:
        by_attack[r["attack"]] = by_attack.get(r["attack"], 0) + 1
    print(f"forged {len(records)} samples -> {args.out}")
    for a, c in sorted(by_attack.items()):
        print(f"  {a}: {c}")


if __name__ == "__main__":
    main()
