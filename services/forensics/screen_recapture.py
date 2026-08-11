from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


class ScreenRecaptureDetector:
    """Detect Moire patterns indicating the document was photographed from a screen."""

    # Calibrated on synthetic renders (genuine max ~0.13) with a conservative
    # margin so a real phone-captured card at ~0.19 is not false-flagged; real
    # captures carry more mid-band energy than clean renders, so the threshold
    # errs toward not rejecting genuine customers (ADR-024). A real-capture
    # validation set (W4/W5) would let this tighten.
    def __init__(self, freq_threshold: float = 0.25) -> None:
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

        # Moire from a screen grid is a few SHARP peaks standing well above the
        # diffuse mid-band energy of print guilloche. Score by peak prominence
        # (top peak vs local mean), not raw density, which cannot separate a
        # spike from broadband texture.
        band_vals = mid_freq[mask > 0]
        band_mean = float(band_vals.mean()) if band_vals.size else 0.0
        band_std = float(band_vals.std()) if band_vals.size else 0.0

        peak_threshold = max(0.55, band_mean + 4.0 * band_std)
        peaks = np.where(mid_freq > peak_threshold)
        n_peaks = len(peaks[0])

        top = float(mid_freq.max())
        prominence = (top - band_mean) / max(band_std, 1e-6)
        # Map prominence (~4 sigma noise floor, ~10+ sigma for a real grid).
        moire_score = float(min(1.0, max(0.0, (prominence - 4.0) / 8.0)))

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
