from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


class ScreenRecaptureDetector:
    """Detect Moire patterns indicating the document was photographed from a screen.

    Three complementary FFT-domain signals:

    1. **Prominence score** (original) — peak-vs-mean in a wide mid-band annulus.
       Strong on period-7 DL/Aadhaar but diluted on PAN where document texture
       raises band variance.
    2. **Radial-ring scan** (W9, ADR-033) — scan narrow concentric rings in the
       Moiré-relevant frequency band (periods ~3.5-16 px, radii 40-180) and
       report the maximum peak-to-ring-mean ratio.  Genuine documents max at
       ~1.40; Moiré concentrates energy in one ring, pushing above 1.55 for
       period-7.
    3. **Combined score** (W11, ADR-035) — normalized sum of prominence and ring
       ratio.  Period-13 Moiré is too weak to exceed either threshold alone but
       elevates both signals simultaneously.  Genuine documents rarely have both
       signals elevated (max ~1.49 on n=180).

    Flag as recaptured when any signal exceeds its threshold.
    """

    RING_R_MIN = 40
    RING_R_MAX = 180
    RING_WIDTH = 3
    RING_STEP = 2

    def __init__(
        self,
        freq_threshold: float = 0.25,
        ring_threshold: float = 1.45,
        combined_threshold: float = 1.55,
    ) -> None:
        self.freq_threshold = freq_threshold
        self.ring_threshold = ring_threshold
        self.combined_threshold = combined_threshold

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
            return {"is_recaptured": False, "moire_score": 0.0,
                    "ring_ratio": 0.0, "combined_score": 0.0,
                    "dominant_frequencies": []}

        magnitude_norm = magnitude / max_mag

        cy, cx = optimal_rows // 2, optimal_cols // 2

        # --- prominence score (wide-band annulus) ---
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
            return {"is_recaptured": False, "moire_score": 0.0,
                    "ring_ratio": 0.0, "combined_score": 0.0,
                    "dominant_frequencies": []}

        band_vals = mid_freq[mask > 0]
        band_mean = float(band_vals.mean()) if band_vals.size else 0.0
        band_std = float(band_vals.std()) if band_vals.size else 0.0

        peak_threshold = max(0.55, band_mean + 4.0 * band_std)
        peaks = np.where(mid_freq > peak_threshold)
        n_peaks = len(peaks[0])

        top = float(mid_freq.max())
        prominence = (top - band_mean) / max(band_std, 1e-6)
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

        # --- radial-ring scan (narrow-band peak/mean ratio) ---
        ring_ratio = self._ring_scan(magnitude, cy, cx)

        combined_score = (moire_score / self.freq_threshold
                          + ring_ratio / self.ring_threshold)

        is_recaptured = (moire_score > self.freq_threshold
                         or ring_ratio > self.ring_threshold
                         or combined_score > self.combined_threshold)

        return {
            "is_recaptured": is_recaptured,
            "moire_score": moire_score,
            "ring_ratio": float(ring_ratio),
            "combined_score": round(float(combined_score), 4),
            "dominant_frequencies": dominant_frequencies,
        }

    def _ring_scan(self, magnitude: np.ndarray, cy: int, cx: int) -> float:
        h, w = magnitude.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt(
            (y_grid.astype(np.float32) - cy) ** 2
            + (x_grid.astype(np.float32) - cx) ** 2
        )

        step = self.RING_STEP
        rmin = self.RING_R_MIN
        ring_idx = ((dist - rmin) / step).astype(np.int32)
        n_rings = (self.RING_R_MAX - rmin) // step
        valid = (ring_idx >= 0) & (ring_idx < n_rings)

        flat_idx = ring_idx[valid]
        flat_mag = magnitude[valid]

        sums = np.bincount(flat_idx, weights=flat_mag, minlength=n_rings)
        counts = np.bincount(flat_idx, minlength=n_rings).astype(np.float64)
        maxes = np.full(n_rings, -np.inf)
        np.maximum.at(maxes, flat_idx, flat_mag)

        best = 0.0
        for i in range(n_rings):
            if counts[i] < 10:
                continue
            ring_mean = sums[i] / counts[i]
            if ring_mean < 1e-6:
                continue
            ratio = float(maxes[i]) / ring_mean
            if ratio > best:
                best = ratio
        return best
