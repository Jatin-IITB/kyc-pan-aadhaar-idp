"""PIL renderers for synthetic Indian identity documents.

Templates are structurally faithful (layout, field labels, security-pattern
style) but deliberately non-counterfeit: emblems are placeholder glyphs and no
real security features are replicated. Every renderer returns the image plus
pixel bounding boxes for each field VALUE, which become YOLO training labels
and extraction ground truth.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import qrcode
from PIL import Image, ImageDraw

from tools.forge.fonts import load_font

CANVAS = (1000, 630)  # credit-card aspect, generous DPI

BBox = List[int]


def _text(draw: ImageDraw.ImageDraw, xy, s: str, font, fill) -> BBox:
    draw.text(xy, s, font=font, fill=fill)
    l, t, r, b = draw.textbbox(xy, s, font=font)
    return [int(l), int(t), int(r), int(b)]


def _guilloche(draw: ImageDraw.ImageDraw, w: int, h: int, rng: np.random.Generator,
               colors=((205, 215, 240), (240, 210, 215))) -> None:
    """Repeating sine-wave security pattern — pale, thin, dense."""
    for band, color in enumerate(colors):
        amp = 10 + int(rng.integers(0, 8))
        period = 60 + int(rng.integers(0, 40))
        phase = float(rng.random() * math.tau)
        for y0 in range(-20, h + 20, 14):
            pts = [
                (x, y0 + amp * math.sin(math.tau * x / period + phase + band))
                for x in range(0, w + 8, 8)
            ]
            draw.line(pts, fill=color, width=1)


def _emblem(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color=(90, 90, 110)) -> None:
    """Placeholder national-emblem glyph: circle with spokes on a plinth."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
    for k in range(12):
        a = math.tau * k / 12
        draw.line(
            [cx + 0.35 * r * math.cos(a), cy + 0.35 * r * math.sin(a),
             cx + 0.92 * r * math.cos(a), cy + 0.92 * r * math.sin(a)],
            fill=color, width=2,
        )
    draw.rectangle([cx - r, cy + r + 4, cx + r, cy + r + 10], fill=color)


def _photo(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    """Synthetic portrait: unique high-texture content per identity."""
    base = np.zeros((h, w, 3), np.uint8)
    for i in range(h):  # vertical studio-backdrop gradient
        shade = 150 + int(60 * i / h)
        base[i, :] = (shade - 20, shade - 8, shade)
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    skin = tuple(int(v) for v in (rng.integers(150, 220), rng.integers(120, 180), rng.integers(100, 150)))
    hair = tuple(int(v) for v in (rng.integers(20, 70),) * 3)
    cx, cy = w // 2, int(h * 0.42)
    fw, fh = int(w * 0.42), int(h * 0.55)
    d.polygon([(int(w * 0.1), h), (cx, int(h * 0.62)), (int(w * 0.9), h)],
              fill=(int(rng.integers(30, 120)), int(rng.integers(30, 90)), int(rng.integers(60, 140))))
    d.ellipse([cx - fw // 2, cy - fh // 2, cx + fw // 2, cy + fh // 2], fill=skin)
    d.arc([cx - fw // 2, cy - fh // 2 - 6, cx + fw // 2, cy + int(fh * 0.25)],
          start=180, end=360, fill=hair, width=int(fh * 0.22))
    ey = cy - int(fh * 0.08)
    for ex in (cx - fw // 5, cx + fw // 5):
        d.ellipse([ex - 4, ey - 3, ex + 4, ey + 3], fill=(30, 30, 30))
    d.line([cx - fw // 8, cy + fh // 5, cx + fw // 8, cy + fh // 5], fill=(90, 50, 50), width=3)
    arr = np.array(img).astype(np.int16)
    arr += rng.integers(-8, 9, arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _signature(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int],
               rng: np.random.Generator) -> None:
    x1, y1, x2, y2 = box
    n = 24
    xs = np.linspace(x1, x2, n)
    ys = (y1 + y2) / 2 + (y2 - y1) * 0.4 * np.sin(
        np.linspace(0, float(rng.integers(3, 6)) * math.pi, n)
    ) * rng.random(n)
    draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(20, 20, 90), width=2)


def _qr(payload: str, size: int) -> Image.Image:
    q = qrcode.QRCode(border=1, box_size=3)
    q.add_data(payload)
    q.make(fit=True)
    return q.make_image(fill_color="black", back_color="white").resize((size, size))


def _finish(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# --- PAN card -------------------------------------------------------------

def render_pan(fields: Dict[str, str], rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, BBox]]:
    w, h = CANVAS
    img = Image.new("RGB", CANVAS, (247, 246, 238))
    d = ImageDraw.Draw(img)
    _guilloche(d, w, h, rng)

    d.rectangle([0, 0, w, 86], fill=(235, 240, 250))
    _emblem(d, w // 2, 42, 26)
    _text(d, (30, 18), "INCOME TAX DEPARTMENT", load_font("bold", 30), (25, 40, 90))
    gt = "GOVT. OF INDIA"
    gw = d.textlength(gt, font=load_font("bold", 30))
    _text(d, (w - 30 - gw, 18), gt, load_font("bold", 30), (25, 40, 90))
    _text(d, (30, 96), "Permanent Account Number Card", load_font("regular", 22), (60, 60, 70))

    boxes: Dict[str, BBox] = {}
    label_f, value_f = load_font("regular", 20), load_font("bold", 30)

    boxes["pan_number"] = _text(d, (60, 170), fields["pan_number"], load_font("mono", 44), (10, 10, 10))

    y = 250
    for key, label in (("name", "Name"), ("father_name", "Father's Name"),
                       ("date_of_birth", "Date of Birth")):
        _text(d, (60, y), label, label_f, (110, 110, 120))
        boxes[key] = _text(d, (60, y + 26), fields[key], value_f, (15, 15, 20))
        y += 92

    img.paste(_photo(rng, 170, 200), (w - 230, 190))
    d.rectangle([w - 230, 190, w - 60, 390], outline=(120, 120, 130), width=2)
    d.rectangle([60, h - 90, 330, h - 40], outline=(160, 160, 170), width=1)
    _signature(d, (75, h - 85, 315, h - 45), rng)
    _text(d, (60, h - 36), "Signature", load_font("regular", 16), (110, 110, 120))
    img.paste(_qr(fields["pan_number"], 120), (w - 190, h - 160))

    return _finish(img), boxes


# --- Aadhaar card ---------------------------------------------------------

def render_aadhaar(fields: Dict[str, str], rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, BBox]]:
    w, h = CANVAS
    img = Image.new("RGB", CANVAS, (252, 252, 252))
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, w, 10], fill=(255, 140, 20))   # saffron
    d.rectangle([0, h - 10, w, h], fill=(20, 130, 60))  # green
    _emblem(d, 60, 52, 24)
    _text(d, (100, 26), "Government of India", load_font("serif", 26), (40, 40, 50))
    _text(d, (100, 56), "Unique Identification Authority", load_font("regular", 18), (120, 120, 130))

    img.paste(_photo(rng, 190, 230), (46, 110))
    d.rectangle([46, 110, 236, 340], outline=(150, 150, 160), width=2)

    boxes: Dict[str, BBox] = {}
    label_f, value_f = load_font("regular", 19), load_font("bold", 27)
    x, y = 270, 120
    for key, label in (("name", "Name"), ("date_of_birth", "DOB"), ("gender", "Gender")):
        _text(d, (x, y), f"{label}:", label_f, (120, 120, 130))
        lw = d.textlength(f"{label}: ", font=label_f)
        boxes[key] = _text(d, (x + lw + 6, y - 4), fields[key], value_f, (20, 20, 25))
        y += 58

    _text(d, (x, y), "Address:", label_f, (120, 120, 130))
    addr = fields["address"]
    lines = [addr[i:i + 44] for i in range(0, min(len(addr), 132), 44)]
    ay, union = y + 26, None
    small = load_font("regular", 20)
    for ln in lines:
        bb = _text(d, (x, ay), ln, small, (45, 45, 55))
        union = bb if union is None else [min(union[0], bb[0]), min(union[1], bb[1]),
                                          max(union[2], bb[2]), max(union[3], bb[3])]
        ay += 26
    boxes["address"] = union or [x, y, x + 1, y + 1]

    num_f = load_font("mono", 46)
    nw = d.textlength(fields["aadhaar_number"], font=num_f)
    boxes["aadhaar_number"] = _text(d, ((w - nw) // 2, h - 130), fields["aadhaar_number"],
                                    num_f, (25, 25, 30))
    d.line([(w - nw) // 2, h - 70, (w + nw) // 2, h - 70], fill=(200, 30, 30), width=3)
    img.paste(_qr(fields["aadhaar_number"], 130), (w - 180, 120))

    return _finish(img), boxes


# --- Driving license ------------------------------------------------------

def render_dl(fields: Dict[str, str], rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, BBox]]:
    w, h = CANVAS
    img = Image.new("RGB", CANVAS, (240, 246, 244))
    d = ImageDraw.Draw(img)
    _guilloche(d, w, h, rng, colors=((205, 235, 220), (215, 225, 245)))

    d.rectangle([0, 0, w, 92], fill=(20, 90, 60))
    _text(d, (30, 14), "INDIAN UNION DRIVING LICENCE", load_font("bold", 30), (245, 245, 240))
    _text(d, (30, 54), "Issued by the Transport Department", load_font("regular", 19), (210, 225, 215))
    _emblem(d, w - 60, 46, 24, color=(235, 235, 225))

    boxes: Dict[str, BBox] = {}
    label_f, value_f = load_font("regular", 19), load_font("bold", 25)

    boxes["dl_number"] = _text(d, (40, 116), fields["dl_number"], load_font("mono", 38), (10, 10, 10))

    img.paste(_photo(rng, 170, 205), (w - 220, 120))
    d.rectangle([w - 220, 120, w - 50, 325], outline=(120, 130, 125), width=2)

    y = 190
    rows = (("name", "Name"), ("date_of_birth", "Date of Birth"), ("blood_group", "Blood Group"),
            ("date_of_issue", "Issue Date"), ("date_of_expiry", "Valid Till"))
    for key, label in rows:
        _text(d, (40, y), f"{label}:", label_f, (90, 100, 95))
        lw = d.textlength(f"{label}: ", font=label_f)
        boxes[key] = _text(d, (40 + lw + 6, y - 3), fields[key], value_f, (15, 15, 20))
        y += 52

    _text(d, (40, y), "Address:", label_f, (90, 100, 95))
    addr = fields["address"]
    lines = [addr[i:i + 52] for i in range(0, min(len(addr), 104), 52)]
    ay, union = y + 26, None
    small = load_font("regular", 19)
    for ln in lines:
        bb = _text(d, (40, ay), ln, small, (40, 45, 42))
        union = bb if union is None else [min(union[0], bb[0]), min(union[1], bb[1]),
                                          max(union[2], bb[2]), max(union[3], bb[3])]
        ay += 25
    boxes["address"] = union or [40, y, 41, y + 1]

    _signature(d, (w - 210, h - 90, w - 70, h - 50), rng)
    _text(d, (w - 210, h - 44), "Holder's Signature", load_font("regular", 15), (100, 105, 100))

    return _finish(img), boxes


RENDERERS = {
    "pan": render_pan,
    "aadhaar": render_aadhaar,
    "driving_license": render_dl,
}
