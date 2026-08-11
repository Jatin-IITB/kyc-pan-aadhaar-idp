"""Capture-condition augmentations for synthetic documents.

v1 policy (ADR-023): photometric augmentations and JPEG re-encoding preserve
bounding boxes exactly; 90-degree rotations transform them. Small-angle
rotation and perspective warp are deferred to the hard-eval split where only
field values (not boxes) are scored.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

BBox = List[int]


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    horizontal = rng.random() < 0.5
    ramp = np.linspace(1.0, float(rng.uniform(0.72, 0.92)), w if horizontal else h)
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    mask = np.tile(ramp, (h, 1)) if horizontal else np.tile(ramp[:, None], (1, w))
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def _rot90(img: np.ndarray, boxes: Dict[str, BBox], k: int) -> Tuple[np.ndarray, Dict[str, BBox]]:
    h, w = img.shape[:2]
    if k == 1:      # 90 clockwise: (x, y) -> (h-1-y, x)
        out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        f = lambda b: [h - b[3], b[0], h - b[1], b[2]]
    elif k == 2:    # 180
        out = cv2.rotate(img, cv2.ROTATE_180)
        f = lambda b: [w - b[2], h - b[3], w - b[0], h - b[1]]
    else:           # 90 counter-clockwise: (x, y) -> (y, w-1-x)
        out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        f = lambda b: [b[1], w - b[2], b[3], w - b[0]]
    return out, {key: f(b) for key, b in boxes.items()}


def augment(
    img: np.ndarray,
    boxes: Dict[str, BBox],
    level: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, BBox], List[str]]:
    """Apply capture-condition noise. Returns (image, boxes, applied-op names)."""
    applied: List[str] = []
    if level == "none":
        return img, boxes, applied

    heavy = level == "full"

    if rng.random() < (0.5 if heavy else 0.3):
        k = int(rng.choice([3, 5] if heavy else [3]))
        img = cv2.GaussianBlur(img, (k, k), 0)
        applied.append(f"blur{k}")

    if rng.random() < 0.6:
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-18, 18))
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        applied.append("brightness")

    if rng.random() < (0.5 if heavy else 0.25):
        img = _shadow(img, rng)
        applied.append("shadow")

    if rng.random() < 0.5:
        sigma = float(rng.uniform(2, 7 if heavy else 4))
        noise = rng.normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        applied.append("noise")

    if heavy and rng.random() < 0.25:
        k = int(rng.choice([1, 2, 3]))
        img, boxes = _rot90(img, boxes, k)
        applied.append(f"rot90x{k}")

    quality = int(rng.integers(55, 91) if heavy else rng.integers(70, 95))
    img = _jpeg(img, quality)
    applied.append(f"jpeg{quality}")

    return img, boxes, applied
