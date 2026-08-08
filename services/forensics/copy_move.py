from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class CopyMoveDetector:
    """Detect duplicated/copy-move regions using DCT-based block matching.

    Uses larger block size (32px) and higher DCT feature dimension to reduce
    false positives from repetitive document backgrounds (watermarks, guilloches).
    Low-texture blocks are filtered by variance threshold.
    """

    def __init__(
        self,
        block_size: int = 32,
        min_matches: int = 8,
        similarity_threshold: float = 0.9997,
        min_block_variance: float = 500.0,
    ) -> None:
        self.block_size = block_size
        self.min_matches = min_matches
        self.similarity_threshold = similarity_threshold
        self.min_block_variance = min_block_variance

    def detect(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if h < self.block_size * 2 or w < self.block_size * 2:
            return {"detected": False, "matched_pairs": [], "confidence": 0.0}

        bs = self.block_size
        blocks: List[Tuple[np.ndarray, int, int]] = []

        step = max(bs // 2, 8)
        for y in range(0, h - bs, step):
            for x in range(0, w - bs, step):
                block = gray[y : y + bs, x : x + bs].astype(np.float32)
                if np.var(block) < self.min_block_variance:
                    continue
                dct_block = cv2.dct(block)
                feature = dct_block[:6, :6].flatten()
                blocks.append((feature, x, y))

        if len(blocks) < 2:
            return {"detected": False, "matched_pairs": [], "confidence": 0.0}

        features = np.array([b[0] for b in blocks])
        positions = [(b[1], b[2]) for b in blocks]

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        features_norm = features / norms

        matched_pairs: List[Dict[str, Any]] = []
        n = len(features_norm)

        sample_size = min(n, 500)
        if n > sample_size:
            indices = np.random.default_rng(42).choice(n, size=sample_size, replace=False)
        else:
            indices = np.arange(n)

        for i in indices:
            sims = features_norm[i] @ features_norm.T
            close = np.where(sims > self.similarity_threshold)[0]
            for j in close:
                if j <= i:
                    continue
                dist = np.sqrt(
                    (positions[i][0] - positions[j][0]) ** 2
                    + (positions[i][1] - positions[j][1]) ** 2
                )
                if dist > bs * 3:
                    matched_pairs.append({
                        "block1": list(positions[i]),
                        "block2": list(positions[j]),
                        "similarity": float(sims[j]),
                    })

        detected = len(matched_pairs) >= self.min_matches
        confidence = min(1.0, len(matched_pairs) / (self.min_matches * 3))

        return {
            "detected": detected,
            "matched_pairs": matched_pairs[:20],
            "confidence": confidence,
        }
