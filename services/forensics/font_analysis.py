from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


class FontConsistencyAnalyzer:
    """Analyze font consistency across text regions using morphological features."""

    def analyze(
        self, image_bgr: np.ndarray, text_regions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if not text_regions:
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            text_regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 20 and h > 10:
                    text_regions.append({"bbox": [x, y, w, h]})

        if len(text_regions) < 2:
            return {"font_consistency_score": 1.0, "inconsistent_regions": []}

        region_metrics: List[Dict[str, Any]] = []
        for region in text_regions:
            bbox = region.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = [int(v) for v in bbox]
            x, y = max(0, x), max(0, y)
            roi = binary[y : y + h, x : x + w]

            if roi.size == 0:
                continue

            eroded = cv2.erode(roi, np.ones((3, 3), np.uint8), iterations=1)
            stroke_pixels = cv2.subtract(roi, eroded)
            stroke_width = float(stroke_pixels.sum()) / max(1, cv2.countNonZero(roi))

            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                roi, connectivity=8
            )
            heights = []
            for i in range(1, n_labels):
                comp_h = stats[i, cv2.CC_STAT_HEIGHT]
                comp_w = stats[i, cv2.CC_STAT_WIDTH]
                if comp_h > 3 and comp_w > 2:
                    heights.append(comp_h)

            avg_height = float(np.mean(heights)) if heights else 0.0

            region_metrics.append({
                "bbox": [x, y, w, h],
                "stroke_width": stroke_width,
                "avg_char_height": avg_height,
                "char_count": len(heights),
            })

        if len(region_metrics) < 2:
            return {"font_consistency_score": 1.0, "inconsistent_regions": []}

        stroke_widths = [m["stroke_width"] for m in region_metrics if m["stroke_width"] > 0]
        char_heights = [m["avg_char_height"] for m in region_metrics if m["avg_char_height"] > 0]

        stroke_cv = float(np.std(stroke_widths) / max(np.mean(stroke_widths), 1e-6)) if stroke_widths else 0.0
        height_cv = float(np.std(char_heights) / max(np.mean(char_heights), 1e-6)) if char_heights else 0.0

        consistency = max(0.0, 1.0 - (stroke_cv + height_cv) / 2)

        inconsistent = []
        if stroke_widths:
            mean_sw = np.mean(stroke_widths)
            std_sw = np.std(stroke_widths)
            for m in region_metrics:
                if abs(m["stroke_width"] - mean_sw) > 2 * std_sw and std_sw > 0:
                    inconsistent.append(m["bbox"])

        return {
            "font_consistency_score": consistency,
            "inconsistent_regions": inconsistent,
        }
