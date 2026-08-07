from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


class ScreenRecaptureDetector:
    """Detect Moire patterns indicating the document was photographed from a screen."""

    def __init__(self, freq_threshold: float = 0.3) -> None:
        self.freq_threshold = freq_threshold

    def detect(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        rows, cols = gray.shape
        optimal_rows = cv2.getOptimalDFTSize(rows)
        optimal_cols = cv2.getOptimalDFTSize(cols)
        padded = np.zeros((optimal_rows, optimal_cols), dtype=np.float32)
        padded[:rows, :cols] = gray

        dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=[0, 1])

        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude = np.log1p(magnitude)

        max_mag = magnitude.max()
        if max_mag == 0:
            return {"is_recaptured": False, "moire_score": 0.0, "dominant_frequencies": []}

        magnitude_norm = magnitude / max_mag

        cy, cx = optimal_rows // 2, optimal_cols // 2
        mask = np.ones_like(magnitude_norm)
        r_low = min(rows, cols) // 20
        cv2.circle(mask, (cx, cy), r_low, 0, -1)

        r_high = min(rows, cols) // 4
        outer_mask = np.zeros_like(magnitude_norm)
        cv2.circle(outer_mask, (cx, cy), r_high, 1, -1)
        mask = mask * outer_mask

        mid_freq = magnitude_norm * mask
        mid_freq_count = np.count_nonzero(mask)

        if mid_freq_count == 0:
            return {"is_recaptured": False, "moire_score": 0.0, "dominant_frequencies": []}

        peak_threshold = 0.7
        peaks = np.where(mid_freq > peak_threshold)
        n_peaks = len(peaks[0])

        peak_density = n_peaks / max(1, mid_freq_count)
        moire_score = min(1.0, peak_density * 100)

        dominant_frequencies: List[Dict[str, Any]] = []
        if n_peaks > 0:
            peak_values = mid_freq[peaks]
            sorted_idx = np.argsort(peak_values)[::-1][:5]
            for idx in sorted_idx:
                py, px = peaks[0][idx], peaks[1][idx]
                freq_y = abs(py - cy) / optimal_rows
                freq_x = abs(px - cx) / optimal_cols
                dominant_frequencies.append({
                    "freq_x": float(freq_x),
                    "freq_y": float(freq_y),
                    "magnitude": float(peak_values[idx]),
                })

        is_recaptured = moire_score > self.freq_threshold

        return {
            "is_recaptured": is_recaptured,
            "moire_score": moire_score,
            "dominant_frequencies": dominant_frequencies,
        }
