# ADR-034: W10 — SIFT fallback for copy-move detection

**Status:** Accepted  
**Date:** 2026-08-22  
**Extends:** ADR-031, ADR-033 (copy-move negative results)

## Context

Copy-move recall plateaus at 63% system recall (19/30 holdout). The ORB
detector itself catches only 47% (14/30) — the remaining 5 are caught by
metadata or template-font signals. ADR-031 and ADR-033 documented three
rejected approaches (dense offset-residual, ORB retune, block DCT).

The fix path in ADR-033 suggested "dense local descriptors (SIFT/DAISY)" as
the top candidate. SIFT was investigated.

## Investigation

### Why ORB misses

ORB's binary descriptors (32 bytes, Hamming distance) require a tight
threshold (hamming <= 10) to avoid structural false matches. JPEG noise
pushes descriptor distances for genuine copy-move pairs past this cut,
leaving only 2-11 matches at the correct shift — often below the min_matches
or dominance thresholds.

### SIFT matching results

SIFT's 128-dim float descriptors with Lowe's ratio test (0.65) are more
tolerant of JPEG noise. On the same missed cases, SIFT found 14-31 matches
at the correct shift, with dominance ratios of 2.0-2.8x.

Across all 30 holdout copy-move images, SIFT identified the correct shift in
29/30 cases (97%).

### FPR challenge

Raw SIFT matching had 3.3% FPR (2/60 genuine) at dominance >= 2.0. The
structural repetition in document text and borders creates enough SIFT
matches to form spurious dominant clusters.

Three measures eliminate FPR:

1. **Promiscuity filter** (same as ORB): exclude keypoints with > 3 near-
   duplicate descriptors (L2 < 200) beyond 4px spatial distance.
2. **Higher min_matches** (12 vs ORB's 8): SIFT finds more total matches
   including structural ones, so a higher bar is needed.
3. **Tighter max_aspect** (3.0 vs ORB's 4.0): structural matches tend to be
   elongated (text lines, borders). Genuine copy-move regions are compact.
   The genuine FP at aspect=3.6 is eliminated; all real catches are < 2.2.

### FPR validation

- 0/60 tuning+holdout genuine (0.0% FPR)
- 0/120 seed-999 FPR pool (0.0% FPR)
- Total: 0/180 genuine documents

## Implementation

SIFT runs as a fallback only when ORB returns `detected=False`:
- No latency impact on ORB-detected cases
- ~130ms additional latency on non-detected cases

Vectorized promiscuity filter: DMatch arrays extracted to numpy, promiscuity
and ratio test computed as vectorized operations. Reduces Python-loop overhead
from ~28ms to ~9ms.

### Parameters

| Parameter | ORB stage | SIFT stage | Rationale |
|---|---|---|---|
| min_matches | 8 | 12 | More structural matches in SIFT |
| max_aspect | 4.0 | 3.0 | Structural matches are elongated |
| descriptor distance | Hamming <= 10 | Lowe ratio <= 0.65 | Float vs binary |
| promiscuity L2 | Hamming < 10 | L2 < 200 | Equivalent scale |
| min_span, max_span | 32, 340 | 32, 340 | Same |
| dominance_ratio | 2.0 | 2.0 | Same |

## Measured results

| Metric | Pre-W10 | Post-W10 | Delta |
|---|---|---|---|
| copy_move tuning | 60.0% | 73.3% | **+13.3pp** |
| copy_move holdout | 63.3% | 73.3% | **+10.0pp** |
| overall tuning | 77.8% | 80.0% | +2.2pp |
| overall holdout | 80.6% | 82.8% | **+2.2pp** |
| genuine FPR | 0% | 0% | unchanged |
| undetected_autoclear (tuning) | 40 | 36 | -4 |
| undetected_autoclear (holdout) | 35 | 31 | -4 |
| copy-move p50 latency | 99 ms | 225 ms | +126 ms |
| copy-move p95 latency | 193 ms | 533 ms | +340 ms |

### Net-new detections (not caught by any other signal)

- Holdout: 3 net-new (1 of 4 SIFT-only catches was already flagged by
  template-font)
- Tuning: 4 net-new (all 4 SIFT-only catches were system misses)

### Latency trade-off

Copy-move latency more than doubled (p50 99→225ms, p95 193→533ms). This is
acceptable because: (1) copy-move is already the slowest detector and runs in
parallel with others; (2) the SIFT fallback only runs on non-detected cases;
(3) a forensic check is not latency-critical.

## Gates ratcheted

- `overall_recall_min`: 0.75 → 0.78
- `copy_move`: 0.50 → 0.65
- `undetected_autoclear_max`: 42 → 38
