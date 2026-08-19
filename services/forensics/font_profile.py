"""Template-conformance font forensics (ADR-030).

`FontConsistencyAnalyzer` asks whether a document is internally consistent.
That is structurally blind to `font_swap`, which re-renders an entire card with
permuted font roles: every value line shifts together, so no region is an
outlier relative to the document's own mean. Measured recall was exactly 0.000.

This module asks a different question — does the document's typography match
the issuing template? Genuine cards have designed structure (labels `regular`,
values `bold`, ID numbers `mono`); the attack permutes those roles.

Envelopes are calibrated per doc type by tools/forge/calibrate_font.py and
stored in config/font_profiles.json. Held-out measurement is recorded there.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "config" / "font_profiles.json"

# Masthead band excluded: templates legitimately set headers in a different
# face from body text (Aadhaar's "Government of India" is serif by design).
# Including it widens the genuine envelope enough to swallow the swap signal.
HEADER_FRACTION = 0.18


def _text_lines(gray: np.ndarray):
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dil = cv2.dilate(binv, ker, iterations=1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 45 and 9 <= h <= 90 and w / h >= 1.6:
            boxes.append((x, y, w, h))
    return binv, boxes


def _line_features(binv: np.ndarray, box) -> Optional[Dict[str, Any]]:
    x, y, w, h = box
    roi = binv[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    ink = cv2.countNonZero(roi)
    if ink < 40:
        return None

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
    ridge = dist[roi > 0]
    if ridge.size == 0:
        return None
    # Serif faces modulate thick/thin within a glyph; bold sans is near-uniform.
    mod = float(ridge.std() / max(ridge.mean(), 1e-6))

    # Serif terminals introduce corners that sans-serif lacks.
    harris = cv2.cornerHarris(roi.astype(np.float32) / 255.0, 2, 3, 0.04)
    peak = harris.max()
    corner = float((harris > 0.01 * peak).sum()) / max(ink, 1) if peak > 0 else 0.0

    n, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    xs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_HEIGHT] >= 5 and stats[i, cv2.CC_STAT_WIDTH] >= 2:
            xs.append(stats[i, cv2.CC_STAT_LEFT])
    adv_cv = None
    if len(xs) >= 5:
        adv = np.diff(np.sort(np.array(xs, dtype=float)))
        adv = adv[adv > 1]
        if len(adv) >= 4:
            # Monospace ID numbers have near-uniform advance; proportional
            # faces do not. Swapping mono away raises this.
            adv_cv = float(adv.std() / max(adv.mean(), 1e-6))

    return {"mod": mod, "corner": corner, "adv_cv": adv_cv, "ink": ink}


def signature(image_bgr: np.ndarray) -> Optional[Dict[str, float]]:
    """Typographic signature of a document, or None when there is too little text."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binv, boxes = _text_lines(gray)
    boxes = [b for b in boxes if b[1] > HEADER_FRACTION * gray.shape[0]]
    feats = [f for f in (_line_features(binv, b) for b in boxes) if f]
    if len(feats) < 3:
        return None

    by_ink = sorted(feats, key=lambda f: -f["ink"])
    top = by_ink[:max(3, len(by_ink) // 3)]
    cvs = [f["adv_cv"] for f in feats if f["adv_cv"] is not None]

    return {
        "corner_top": float(np.mean([f["corner"] for f in top])),
        "mod_top": float(np.mean([f["mod"] for f in top])),
        "adv_cv_min": float(min(cvs)) if cvs else None,
    }


class TemplateFontForensics:
    """Flags typography that deviates from a calibrated per-doc-type envelope."""

    def __init__(self, profile_path: Path | str = PROFILE_PATH):
        self.profiles: Dict[str, Any] = {}
        self.vote = 1
        self.meta: Dict[str, Any] = {}
        p = Path(profile_path)
        if not p.exists():
            logger.info("No font profile at %s — template font forensics disabled", p)
            return
        try:
            blob = json.loads(p.read_text())
            self.profiles = blob.get("profiles", {})
            self.vote = int(blob.get("vote", 1))
            self.meta = blob.get("measured_holdout", {})
        except Exception:
            logger.warning("Could not read font profile at %s — disabled", p)

    @property
    def available(self) -> bool:
        return bool(self.profiles)

    def analyze(self, image_bgr: np.ndarray, doc_type: str) -> Dict[str, Any]:
        prof = self.profiles.get((doc_type or "").lower())
        if not prof:
            return {"template_mismatch": False, "reason": "no profile for doc type"}

        sig = signature(image_bgr)
        if sig is None:
            return {"template_mismatch": False, "reason": "insufficient text"}

        breaches: List[Dict[str, Any]] = []
        for feat, spec in prof.items():
            v = sig.get(feat)
            if v is None:
                continue
            bound, side = spec["bound"], spec["side"]
            if (side == "high" and v > bound) or (side == "low" and v < bound):
                breaches.append({"feature": feat, "value": round(v, 4),
                                 "bound": round(bound, 4), "side": side})

        mismatch = len(breaches) >= self.vote
        # Strength saturates at 2 breaches beyond the vote threshold; a single
        # marginal breach should not read as strongly as a broad mismatch.
        strength = min(1.0, len(breaches) / max(self.vote + 2, 1)) if mismatch else 0.0
        return {
            "template_mismatch": mismatch,
            "strength": round(strength, 3),
            "breaches": breaches,
            "signature": {k: (round(v, 4) if v is not None else None)
                          for k, v in sig.items()},
        }
