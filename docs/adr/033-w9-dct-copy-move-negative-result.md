# ADR-033: W9 — DCT copy-move rejected, radial-ring Moiré accepted

**Status:** Accepted (screen recapture), Rejected (copy-move DCT)  
**Date:** 2026-08-20  
**Extends:** ADR-031 (copy-move negative results)

## Context

Copy-move recall plateaus at 63% (holdout). ADR-031 identified ORB's failure
mode: `not_dominant` rejection on ~40% of forgeries where the duplicated photo
region produces too few keypoints relative to the document's periodic text
structure. The fix path suggested was region descriptors (Zernike/polar-cosine).

W9 attempted a different approach: block DCT matching as a fallback when ORB
fails. The hypothesis was that smooth photo regions (faces, skin, clothing)
where ORB finds insufficient keypoints would have distinctive DCT spectra that
survive copy-move duplication.

## Implementation

Block DCT matching stage (`_dct_stage`):
- 16x16 blocks at step 8 (overlapping)
- 12 lowest-frequency AC coefficients (zigzag order) as block descriptor
- AC energy filter (threshold 100.0) to exclude flat/uniform blocks
- Quantized hash grouping (quant=6) for exact matching
- Same dominance ratio (2.0) and region-span guards as ORB stage

## Results

### FPR validation (ADR-031 requirement)
- 0/120 genuine documents produced false positives (0.0% FPR)
- Validated on seed 999, 40 per doc type

### Recall
- **Zero additional detections.** DCT caught no cases that ORB missed.
- Tuning: 60% (unchanged), Holdout: 63.3% (unchanged)

### Latency impact
- Added ~80ms per non-detected case (ORB+DCT sequential)
- Genuine doc p50: 421ms (vs ~99ms ORB-only)

## Root cause analysis

### Finding 1: Grid alignment kills block matching

The forgery offset is rarely a multiple of the step size. With step=8, only
1/8 of offsets align. A forgery at offset (-175, 94) has sub-step remainders
of (1, 6) pixels — enough to completely change the DCT coefficients of each
block. The source block and its copy never produce the same quantized hash.

**This is the same fundamental limitation ADR-031 documented for the v2 DCT
detector (step=16).** Reducing the step from 16 to 8 improved the alignment
probability from 6.25% to 12.5%, but 87.5% failure rate is still fatal.

Step=1 would solve alignment but produces 64x more blocks (~600K per image),
making the approach impractical.

### Finding 2: Periodic structure dominates shift clustering

Even when blocks do match (via structural similarity), the document's line
spacing creates thousands of shift matches at multiples of ~56px vertical.
Top shift: count=3468; runner-up: count=2982; dominance ratio=1.2 (required
2.0). The forgery's shift cluster, if it existed, would be buried under
structural periodicity.

This is the same dominance-failure mechanism that defeats ORB on these cases.
The approach differs in HOW it finds matches (hash vs. Hamming distance) but
the structural interference is identical.

### Finding 3: JPEG quantization noise breaks exact matching

JPEG compression quantizes each 8x8 DCT block independently. After save/load,
the 16x16 blocks spanning multiple JPEG blocks accumulate quantization noise
that shifts the AC coefficients beyond the hash quantization tolerance (quant=6).
Coarser quantization (larger quant) would increase tolerance but also increase
structural false matches.

## Decision

**Rejected.** The block DCT approach was reverted. It adds latency (80ms
per miss) with zero recall improvement due to three independent failure modes
that compound: grid misalignment (87.5%), structural dominance, and JPEG noise.

## Updated fix path

The ADR-031 conclusion stands: grid-free approaches are necessary. Candidates,
in order of expected viability:

1. **Dense local descriptors (SIFT/DAISY)** — grid-free, floating-point distance
   matching with tolerance (not exact hash). SIFT is patented but free in
   OpenCV 4.x. Would need the same dominance/span guards.
2. **Learned copy-move localizer** — a small CNN trained to predict a per-pixel
   forgery mask. Out of scope for the current project.
3. **Phase correlation** — FFT-based global shift detection. Fast but finds
   only ONE shift, dominated by structural periodicity.

The honest assessment: copy-move at 63% may be near the ceiling achievable
with unsupervised feature matching on synthetic documents with periodic
structure. A meaningful jump likely requires either (a) dense descriptors
that tolerate JPEG noise, or (b) restricting the search to photo regions
(requires YOLO field detection).

---

## Part 2: Radial-ring Moiré scan (accepted)

### Context

Screen recapture detection was at 43% holdout (13/30). The existing wide-band
FFT prominence score works well for period-7 Moiré on DL/Aadhaar but fails on
PAN cards where document texture raises band variance, diluting prominence below
the threshold (0.25). Period-13 and period-3 cases also missed.

Diagnosis showed all period-7 PAN misses had scores 0.158–0.219 (just below
threshold). The peak IS there — it's just not prominent enough relative to the
broadband texture.

### Approach: narrow radial-ring scan

Instead of computing peak prominence over a wide annulus (r_low to r_high),
scan narrow concentric rings (width=3, step=2) within the Moiré-relevant
frequency band (radii 40–180, corresponding to periods ~3.5–16 px). For each
ring, compute peak-to-mean ratio.

Moiré concentrates FFT energy in one ring at the Moiré spatial frequency. In a
narrow ring, the noise floor (document texture) is much lower than in a wide
annulus, so the signal-to-noise is higher. Genuine documents have smooth radial
energy distribution: no ring has a notably higher peak/mean ratio than others.

Excluding radii > 180 (periods < 3.5 px) is critical: genuine documents have
structural energy at period ~3 (print grain, text edges) with ratios up to 1.47.
Genuine ring ratios in the Moiré band (r=40–180) max at 1.39.

### Implementation

- `_ring_scan()` added to `ScreenRecaptureDetector`
- Vectorized: single distance computation + bincount for sums/maxes
- Combined detection: flag if `moire_score > 0.25` OR `ring_ratio > 1.45`
- Latency: ~24ms p50 (was ~16ms), +8ms from ring scan

### Threshold selection

Genuine max ring_ratio: 1.3935 (n=120, seed 999, all doc types).
Threshold: 1.45. Margin: 0.057.

Following interior-point selection: the first value clearing 0% FPR is ~1.40;
1.45 provides a comfortable margin. Validated at 0/120 genuine FPR.

### Measured results

| Metric | Pre-W9 | Post-W9 | Delta |
|---|---|---|---|
| screen_recapture tuning | 33.3% | 53.3% | **+20.0pp** |
| screen_recapture holdout | 43.3% | 70.0% | **+26.7pp** |
| overall tuning | 74.4% | 77.8% | +3.4pp |
| overall holdout | 76.1% | 80.6% | **+4.5pp** |
| genuine FPR | 0% | 0% | unchanged |
| undetected_autoclear (tuning) | 46 | 40 | -6 |
| undetected_autoclear (holdout) | 43 | 35 | -8 |

Per-period holdout breakdown:
- Period 7 (high sev): 8/10 → **10/10** (100%)
- Period 13 (med sev): 2/10 → 3/10
- Period 3 (low sev): 3/10 → 7/10 (some caught at harmonics)

### Gates ratcheted

- `overall_recall_min`: 0.70 → 0.75
- `screen_recapture`: 0.30 → 0.50
- `undetected_autoclear_max`: 50 → 42
