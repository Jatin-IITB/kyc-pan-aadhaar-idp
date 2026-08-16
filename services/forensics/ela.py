from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


class ELADetector:
    """Error Level Analysis — detect tampered regions by re-compression difference."""

    def __init__(self, quality: int = 90, threshold: float = 0.11) -> None:
        self.quality = quality
        self.threshold = threshold

    def analyze(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        success, encoded = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        if not success:
            return {"ela_score": 0.0, "suspicious_regions": []}

        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(image_bgr, recompressed)

        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        max_val = gray_diff.max()
        if max_val == 0:
            return {"ela_score": 0.0, "suspicious_regions": []}

        normalized = (gray_diff.astype(np.float32) / 255.0)
        ela_score = float(normalized.mean())

        suspicious_regions: List[List[int]] = []

        # Method 1: fixed-threshold contours (gross tampering).
        _, binary = cv2.threshold(
            gray_diff, int(255 * self.threshold), 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                suspicious_regions.append([x, y, w, h])

        return {
            "ela_score": ela_score,
            "suspicious_regions": suspicious_regions,
        }
