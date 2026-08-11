from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class CopyMoveDetector:
    """Detect duplicated regions (copy-move forgery) via DCT block matching
    with shift-vector clustering.

    Genuine identity documents are dense with repeating print structure
    (guilloche lines, watermarks, tiled emblems) that produces pixel-identical
    blocks by design. Three guards separate that from actual forgery (ADR-022):

    1. DC removal — the DC coefficient is zeroed in block features so shared
       brightness can never drive similarity.
    2. Repetition filter — a block matching more than ``max_neighbor_matches``
       distant blocks belongs to a repeating texture and is discarded; a real
       copy-move source/destination pair matches once.
    3. Shift-vector clustering — a duplicated region moves by one offset, so
       forgery concentrates matched pairs in a single displacement bin.
       ``min_matches`` is required in the dominant bin, not image-wide.
    """

    def __init__(
        self,
        block_size: int = 32,
        min_matches: int = 8,
        similarity_threshold: float = 0.9997,
        min_block_variance: float = 500.0,
        max_neighbor_matches: int = 2,
        shift_bin_px: int = 8,
        max_blocks: int = 4096,
    ) -> None:
        self.block_size = block_size
        self.min_matches = min_matches
        self.similarity_threshold = similarity_threshold
        self.min_block_variance = min_block_variance
        self.max_neighbor_matches = max_neighbor_matches
        self.shift_bin_px = shift_bin_px
        self.max_blocks = max_blocks

    def detect(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        empty: Dict[str, Any] = {
            "detected": False,
            "matched_pairs": [],
            "confidence": 0.0,
            "dominant_shift": None,
        }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < self.block_size * 2 or w < self.block_size * 2:
            return empty

        bs = self.block_size
        step = max(bs // 2, 8)
        feats: List[np.ndarray] = []
        pos_list: List[Tuple[int, int]] = []
        for y in range(0, h - bs, step):
            for x in range(0, w - bs, step):
                block = gray[y : y + bs, x : x + bs].astype(np.float32)
                if np.var(block) < self.min_block_variance:
                    continue
                dct_block = cv2.dct(block)
                dct_block[0, 0] = 0.0
                feats.append(dct_block[:6, :6].flatten())
                pos_list.append((x, y))

        n = len(feats)
        if n < 2:
            return empty

        features = np.asarray(feats, dtype=np.float32)
        positions = np.asarray(pos_list, dtype=np.float32)

        if n > self.max_blocks:
            keep = np.random.default_rng(42).choice(n, size=self.max_blocks, replace=False)
            features = features[keep]
            positions = positions[keep]
            n = self.max_blocks

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        fn = features / norms

        sim = fn @ fn.T
        cand = np.argwhere(np.triu(sim > self.similarity_threshold, k=1))
        if cand.size == 0:
            return empty

        spans = positions[cand[:, 1]] - positions[cand[:, 0]]
        far = np.hypot(spans[:, 0], spans[:, 1]) > bs * 3
        cand, spans = cand[far], spans[far]
        if cand.size == 0:
            return empty

        # Repetition filter: promiscuous blocks are texture, not forgery.
        counts = np.bincount(cand.ravel(), minlength=n)
        singular = (counts[cand[:, 0]] <= self.max_neighbor_matches) & (
            counts[cand[:, 1]] <= self.max_neighbor_matches
        )
        cand, spans = cand[singular], spans[singular]
        if cand.size == 0:
            return empty

        # Shift-vector clustering: one forgery = one displacement.
        canon = spans.copy()
        flip = (canon[:, 0] < 0) | ((canon[:, 0] == 0) & (canon[:, 1] < 0))
        canon[flip] *= -1
        bins = np.round(canon / self.shift_bin_px).astype(int)
        tally = Counter(map(tuple, bins))
        dom_bin, dom_count = tally.most_common(1)[0]

        in_bin = np.all(bins == np.asarray(dom_bin), axis=1)
        matched_pairs = [
            {
                "block1": [int(positions[i][0]), int(positions[i][1])],
                "block2": [int(positions[j][0]), int(positions[j][1])],
                "similarity": float(sim[i, j]),
            }
            for i, j in cand[in_bin][:20]
        ]

        detected = dom_count >= self.min_matches
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
