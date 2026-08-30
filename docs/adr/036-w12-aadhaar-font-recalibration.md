# ADR-036: W12 — Aadhaar font forensics recalibration

**Status:** Accepted
**Date:** 2026-08-27
**Extends:** ADR-030, ADR-032 (template-font forensics)

## Context

After W8 (ADR-032), font_swap holdout recall was 57% (17/30 system), but the
per-doc-type breakdown revealed a critical gap:

| Doc type | Template-font catches | System catches |
|----------|----------------------|----------------|
| PAN      | 9/10                 | 9/10           |
| DL       | 6/10                 | 8/10           |
| Aadhaar  | **0/10**             | 0/10           |

Aadhaar font_swap was completely undetected. This was flagged as the #1
remaining gap in ADR-030 L1: "Aadhaar recall is ~6%... its number-dominant
layout needs an ID-line-specific feature."

## Investigation

### Why the old profile failed

The Aadhaar profile used `margin=8` (median ± 8×MAD) for bounds on 3
features (corner_top, mod_top, adv_cv_min). With vote=1, any single breach
triggers. But at margin=8, the bounds were too wide to discriminate:

| Feature | Genuine range | margin=8 bound | Swapped range | Catches |
|---------|--------------|----------------|---------------|---------|
| corner_top (high) | [0.10, 0.44] | 0.63 | [0.10, 0.53] | 0 |
| mod_top (low)     | [0.35, 0.57] | 0.30 | [0.32, 0.57] | 0 |
| adv_cv_min (high) | [0.15, 0.23] | 0.23 | [0.10, 0.23] | 0 |

The margin pushed bounds far beyond genuine extremes, swallowing the swap
signal entirely.

### Feature analysis

Compared n=30 genuine and n=20 font-swapped Aadhaar signatures. Per-image
analysis showed which features fell outside the genuine range for each
swapped image:

- `corner_top` high: 4/20 swapped outside range
- `mod_top` low: 4/20 swapped outside range
- `adv_cv_min`: 3/20 (2 low, 1 high)
- **`id_width_cv`**: **9/20** (6 low, 3 high) — best single-feature discrimination

`id_width_cv` measures glyph width uniformity on the ID-number line. Aadhaar's
12-digit number has a tight genuine cluster (CV 0.15-0.25). Font swaps scatter
both directions: overly uniform faces compress (CV 0.05) and proportional
faces expand (CV 0.28).

### Margin sweep

Tested margin values on the 3 existing features (no id_width_cv):

| margin | FPR (n=32 genuine) | Recall (n=32 swapped) |
|--------|--------------------|-----------------------|
| 3      | 0%                 | 31%                   |
| 4      | 0%                 | 25%                   |
| 5      | 0%                 | 19%                   |
| 8 (old)| 0%                 | 6%                    |

Adding `id_width_cv` as a low-only feature (catches swap faces with
unusually uniform ID-line glyphs):

| margin | FPR | Recall | LOO FPR |
|--------|-----|--------|---------|
| 3 + idw low | 3% (1 FP) | 81% | 6.2% |
| **4 + idw low** | **0%** | **72%** | **0%** |
| 5 + idw low | 0% | 56% | 0% |

**margin=4 with id_width_cv low** is the clear winner: 72% recall at 0% FPR
and 0% LOO FPR. Margin=3 was rejected due to 6.2% LOO FPR.

### Why low-only (not band)?

Band (two-sided) bounds were investigated first but rejected:
- The genuine id_width_cv distribution is right-skewed (tight core 0.15-0.21
  with occasional outliers to 0.29)
- MAD-based band at margin=8 produces high bound at 0.263 — two genuine
  holdout images at 0.291 and 0.292 breach it → 12.5% FPR
- Widening the band to cover outliers (margin≥12) makes the low side too
  permissive to catch anything

A low-only bound avoids the right-tail FPR entirely. The 3 high-side swapped
images (id_width_cv > 0.25) are recoverable by corner_top/mod_top tightening.

## Implementation

Changes to `tools/forge/calibrate_font.py`:
- `DT_MARGIN["aadhaar"]`: 8.0 → 4.0
- `DT_EXTRA["aadhaar"]`: `{"id_width_cv": "low"}`

Changes to `services/forensics/font_profile.py`:
- Added `side: "band"` support (backward-compatible; not currently used but
  available for future features)

Profile re-emitted via `tools/forge/calibrate_font.py --n 16 --emit --force`.

### Calibration holdout results (seeds 9001/24601)

| Doc type | Pre-W12 recall | Post-W12 recall | FPR |
|----------|---------------|-----------------|-----|
| PAN      | 81.2%         | 81.2%           | 0%  |
| Aadhaar  | **0%**        | **62.5%**       | 0%  |
| DL       | 93.8%         | 93.8%           | 0%  |
| Overall  | 54.2%*        | 79.2%           | 0%  |

\* Previous Aadhaar profile at margin=8 contributed near-zero recall

## Measured results (eval harness, seeds 42/123 and 777/888)

| Metric | Pre-W12 | Post-W12 | Delta |
|--------|---------|----------|-------|
| font_swap tuning | 53.3% | 66.7% | **+13.3pp** |
| font_swap holdout | 56.7% | 73.3% | **+16.7pp** |
| overall tuning | 83.9% | 86.1% | +2.2pp |
| overall holdout | 86.1% | 88.9% | **+2.8pp** |
| genuine FPR | 0% | 0% | unchanged |
| undetected_autoclear (tuning) | 29 | 25 | -4 |
| undetected_autoclear (holdout) | 25 | 20 | -5 |

## Gates ratcheted

- `overall_recall_min`: 0.80 → 0.84
- `font_swap`: 0.47 → 0.60
- `undetected_autoclear_max`: 32 → 28

## Remaining font_swap gaps

Post-W12, font_swap misses 8/30 holdout:
- PAN: 1 miss (swap that doesn't change typography enough for any feature)
- Aadhaar: ~4 misses (swap effects within tightened but still wide envelope)
- DL: ~3 misses (need vote=2 so single-feature breaches don't trigger)

Further Aadhaar improvement would require either:
1. A tighter margin (margin=3 has 6.2% LOO FPR — unsafe at vote=1)
2. Additional features (stroke-width histogram, serif detection)
3. A different detection architecture (learned discriminator)
