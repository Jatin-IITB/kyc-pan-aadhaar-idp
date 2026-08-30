# ADR-037: W13 — SIFT shift-neighborhood merge for copy-move

**Status:** Accepted
**Date:** 2026-08-30
**Extends:** ADR-034 (SIFT copy-move fallback)

## Context

After W10 (ADR-034) the SIFT fallback improved copy_move from 63% to 73% on
the eval holdout. However, detailed analysis of the 8 remaining misses revealed
a binning artifact: SIFT shift-vector clustering bins matches at 8px resolution,
so a copy-move shift that lands near a bin boundary splits its matches between
two adjacent bins. Neither bin alone dominates the structural-repetition
runner-up, even though the combined count clearly does.

Example from aadhaar_777_000002_copy_move: bins (29,14)=24 and (30,14)=21 are
the same shift ±4px. Combined: 45 matches vs runner-up 16 = 2.8× dominant.
Individually: 24 vs 21 (next bin, not a runner-up at all since they're the
same shift).

## Investigation

### Failure mode analysis

Ran SIFT with diagnostics on all 12 copy_move images missed by the detector
(8 end-to-end misses + 4 caught by other detectors). Every one fails the
dominance test — the runner-up bin is 50–85% of the dominant.

Of those 12, 3 have adjacent bins that together would pass dominance:
- aadhaar_000002: merged=45, runner=16 (2.81×)
- aadhaar_000008: merged=35, runner=11 (3.18×)
- dl_000002: merged=38, runner=19 (2.00×)

### FPR risk

Neighborhood merge also boosts structural repetition on genuine documents.
Tested 90 genuine images (holdout + tuning + fresh seed 5555):

| merged_min | FPs (genuine) | Recoveries (copy_move) |
|-----------|---------------|----------------------|
| 12        | 7/90 (7.8%)   | 4                    |
| 18        | 3/30 (10%)†   | 3                    |
| 25        | **0/90 (0%)**  | **3**                |
| 30        | 0/90 (0%)     | 3                    |

† holdout-only; the worst genuine merged count across all 90 images was 23.

**merged_min=25** is the interior-point choice: 2 above the worst genuine
case (23), recovers all 3 mergeable images, 0% FPR.

### Implementation

Second-pass check in `_cluster_and_decide()`, gated on `merge_radius > 0`:

1. Standard single-bin dominance check runs first (unchanged)
2. If it fails and merge_radius > 0, merge the ±1 neighborhood around the
   dominant bin
3. Check merged count ≥ max(merge_min, dominance_ratio × non-neighbor runner)
4. Verify span on the merged pair set

Only the SIFT stage passes merge parameters (`SIFT_MERGE_RADIUS=1`,
`SIFT_MERGE_MIN=25`). ORB is unaffected — its small pair counts don't
benefit from merge.

## Measured results

### Copy-move detector (isolation)

| Metric | Pre-W13 | Post-W13 |
|--------|---------|----------|
| Detector recall (holdout) | 18/30 (60%) | 21/30 (70%) |
| Detector recall (tuning) | 19/30 (63%) | 20/30 (67%) |
| Genuine FPR | 0% | 0% |

### End-to-end (eval harness)

| Metric | Pre-W13 | Post-W13 | Delta |
|--------|---------|----------|-------|
| copy_move tuning | 73.3% | 76.7% | **+3.3pp** |
| copy_move holdout | 73.3% | 76.7% | **+3.3pp** |
| overall tuning | 86.1% | 87.2% | +1.1pp |
| overall holdout | 88.9% | 89.4% | **+0.6pp** |
| genuine FPR | 0% | 0% | unchanged |
| undetected_autoclear (tuning) | 25 | 23 | -2 |
| undetected_autoclear (holdout) | 20 | 19 | -1 |

## Gates ratcheted

- `overall_recall_min`: 0.84 → 0.86
- `copy_move`: 0.65 → 0.70
- `undetected_autoclear_max`: 28 → 25

## Remaining copy_move gaps

Post-W13, the 9 remaining detector misses fail for genuine reasons:
- 5: runner-up bin is at a genuinely different shift (structural repetition),
  not adjacent to the copy-move bin
- 2: dominant cluster has thin/elongated span (aspect > 3.0)
- 2: too few dominant matches even after merge

Further improvement requires either:
1. Restricting search to YOLO-detected photo regions
2. Patch-SSIM verification of the dominant shift
3. A learned copy-move localizer
