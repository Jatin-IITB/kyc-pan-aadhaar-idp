from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import cv2
import numpy as np


class CopyMoveDetector:
    """Detect duplicated regions (copy-move forgery) via ORB keypoint
    self-matching with shift-vector clustering.

    v3 (ADR-026). The v2 DCT-grid detector sampled blocks on a fixed 16 px
    stride, so a duplication whose offset was not a multiple of the stride
    never produced comparable blocks — the audit measured blindness at ~15/16
    of arbitrary offsets. ORB keypoints are anchored to image content, not to
    a sampling grid, so matching is alignment-free (and tolerant of the mild
    pixel noise a JPEG re-save introduces).

    Three guards separate forgery from the structures genuine documents
    legitimately repeat:

    1. Repetition filter — a keypoint whose descriptor has more than
       ``max_neighbor_matches`` near-duplicates ANYWHERE in the image belongs
       to repeating texture (guilloche, QR modules, tiled emblems) and is
       excluded entirely. Measured before any spatial filtering so periodic
       patterns cannot hide behind a minimum-shift cut.
    2. Shift-vector clustering with a dominance test — a duplicated region
       moves by ONE offset, so forgery concentrates in a single displacement
       bin (``min_matches`` required) that clearly dominates the runner-up
       (``dominance_ratio``). Periodic structure that survives the repetition
       filter (ORB's keypoint budget can sample a sparse subset of a texture)
       spreads across many comparable bins and fails dominance.
    3. Region-span band — the dominant cluster's source keypoints must span
       a REGION: at least ``min_span_px`` on both axes (rejecting thin repeated
       text strips and tiny QR motifs) and at most ``max_span_px`` on both
       (rejecting periodic print structure, whose same-offset matches scatter
       card-wide instead of clustering into one pasted region). A genuine
       photo/field duplication is compact and 2-D; a guilloche or a monospace
       number's repeats are either 1-D or card-spanning.
       Deliberate trade-off: single-line text duplication is out of scope here
       (that is the text-splice attack surface).
    """

    def __init__(
        self,
        min_matches: int = 8,
        # A true copy-move is an exact pixel copy: ORB descriptors match at
        # near-zero Hamming distance (JPEG noise aside). Repeated-but-different
        # text glyphs (a monospace number, a multi-line address) match only
        # loosely, so a tight threshold separates duplication from repetition.
        match_max_hamming: int = 10,
        min_shift_px: int = 48,
        max_neighbor_matches: int = 3,
        shift_bin_px: int = 8,
        min_span_px: float = 32.0,
        max_span_px: float = 340.0,
        max_aspect: float = 4.0,
        dominance_ratio: float = 2.0,
        max_keypoints: int = 5000,
        fast_threshold: int = 5,
    ) -> None:
        self.min_matches = min_matches
        self.match_max_hamming = match_max_hamming
        self.min_shift_px = min_shift_px
        self.max_neighbor_matches = max_neighbor_matches
        self.shift_bin_px = shift_bin_px
        self.min_span_px = min_span_px
        self.max_span_px = max_span_px
        self.max_aspect = max_aspect
        self.dominance_ratio = dominance_ratio
        self.max_keypoints = max_keypoints
        self.fast_threshold = fast_threshold

    def detect(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        empty: Dict[str, Any] = {
            "detected": False,
            "matched_pairs": [],
            "confidence": 0.0,
            "dominant_shift": None,
        }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # A low FAST threshold keeps softer keypoints (photo/portrait texture),
        # which the card's strong text/QR corners would otherwise crowd out of
        # the keypoint budget — leaving a duplicated photo invisible.
        orb = cv2.ORB_create(nfeatures=self.max_keypoints, fastThreshold=self.fast_threshold)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.min_matches * 2:
            return empty

        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        n = len(keypoints)

        # k is a measurement window: promiscuity must be countable up to well
        # past the exclusion threshold, or truncation hides repeating texture.
        # +1 compensates for the self-match slot (knnMatch includes identity).
        k = min(self.max_neighbor_matches + 6, n)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(descriptors, descriptors, k=k)

        # Pass 1 — promiscuity, BEFORE spatial cuts: count near-duplicate
        # descriptors anywhere beyond a few pixels (so a 32 px-period texture
        # is counted even though pairing below would reject those distances).
        near_dups = np.zeros(n, dtype=int)
        for matches in knn:
            for m in matches:
                if m.queryIdx == m.trainIdx or m.distance > self.match_max_hamming:
                    continue
                d = pts[m.queryIdx] - pts[m.trainIdx]
                if float(np.hypot(d[0], d[1])) > 4.0:
                    near_dups[m.queryIdx] += 1
        keep = near_dups <= self.max_neighbor_matches

        # Pass 2 — candidate pairs among non-promiscuous keypoints.
        seen = set()
        pairs: List[tuple] = []
        for matches in knn:
            for m in matches:
                i, j = m.queryIdx, m.trainIdx
                if i == j or not (keep[i] and keep[j]):
                    continue
                if m.distance > self.match_max_hamming:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                d = pts[j] - pts[i]
                if float(np.hypot(d[0], d[1])) < self.min_shift_px:
                    continue
                seen.add(key)
                pairs.append((key[0], key[1], m.distance))
        if not pairs:
            return empty

        # Shift-vector clustering: one forgery = one displacement. Pairs are
        # re-oriented along the canonical displacement so the "source" side is
        # one spatially coherent cluster (required for a meaningful span).
        bins: List[tuple] = []
        oriented: List[tuple] = []
        for i, j, dist in pairs:
            dx, dy = pts[j] - pts[i]
            if dx < 0 or (dx == 0 and dy < 0):
                i, j = j, i
                dx, dy = -dx, -dy
            bins.append((round(dx / self.shift_bin_px), round(dy / self.shift_bin_px)))
            oriented.append((i, j, dist))
        tally = Counter(bins)
        top = tally.most_common(2)
        dom_bin, dom_count = top[0]
        runner_up = top[1][1] if len(top) > 1 else 0
        dominant = dom_count >= max(self.min_matches, self.dominance_ratio * runner_up)
        dom_pairs = [p for p, b in zip(oriented, bins) if b == dom_bin]

        # Region-shape gate: a pasted region is compact, 2-D, and roughly
        # square. Three ways the source cluster betrays repeated print instead:
        #   - min extent tiny        -> a thin repeated text line
        #   - max extent card-wide   -> periodic structure, not one region
        #   - aspect ratio elongated -> a monospace number / date strip
        src = np.array([pts[i] for i, _, _ in dom_pairs])
        extents = src.max(axis=0) - src.min(axis=0) if len(src) > 1 else np.zeros(2)
        min_ext, max_ext = float(min(extents)), float(max(extents))
        aspect = max_ext / max(min_ext, 1.0)
        span_ok = (self.min_span_px <= min_ext
                   and max_ext <= self.max_span_px
                   and aspect <= self.max_aspect)

        matched_pairs = [
            {
                "block1": [int(pts[i][0]), int(pts[i][1])],
                "block2": [int(pts[j][0]), int(pts[j][1])],
                "similarity": round(1.0 - dist / 256.0, 4),
            }
            for i, j, dist in dom_pairs[:20]
        ]

        detected = dominant and span_ok
        confidence = min(1.0, dom_count / (self.min_matches * 2))

        return {
            "detected": detected,
            "matched_pairs": matched_pairs,
            "confidence": confidence if detected else min(confidence, 0.3),
            "dominant_shift": [
                int(dom_bin[0] * self.shift_bin_px),
                int(dom_bin[1] * self.shift_bin_px),
            ],
        }
