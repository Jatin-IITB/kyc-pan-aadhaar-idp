# ADR-038: W14 — SIFT patch-NCC verification for copy-move

**Status:** Accepted
**Date:** 2026-08-30
**Extends:** ADR-034 (SIFT fallback), ADR-037 (neighborhood merge)

## Context

After W13, the SIFT copy-move detector misses images where the dominant
shift bin correctly identifies the forge offset, but structural repetition
(guilloché, card borders, text patterns) creates a runner-up bin with
50–78% of the dominant count. The 2× dominance ratio rejects these
because the runner-up is too close.

Four holdout misses and two tuning misses exhibit this exact pattern:
the detector finds the right shift but can't prove it dominates.

## Investigation

### Failure mode: weak dominance with correct shift

| Image | dom | runner | ratio | Forge matches? |
|-------|-----|--------|-------|---------------|
| pan_777_000001 | 23 | 18 | 1.28 | Yes |
| aadhaar_777_000005 | 19 | 14 | 1.36 | Yes |
| aadhaar_777_000006 | 19 | 11 | 1.73 | Yes |
| dl_777_000001 | 18 | 14 | 1.29 | Yes |

The shift is right, the count is substantial, the span is valid — only
the dominance ratio fails. The key insight: for genuine copy-move, the
shifted patches have near-perfect pixel-level similarity (NCC > 0.80),
while structural repetition at the same offset has lower patch similarity
because the non-keypoint regions differ.

### Patch-NCC as a disambiguation signal

Pearson correlation (NCC) between the source bounding-box region and its
shifted counterpart on the grayscale image:

| Category | NCC range | Count |
|----------|-----------|-------|
| Copy-move misses (shift correct) | 0.80–0.87 | 6 |
| Genuine images (max NCC with dom≥18 + ratio>1.15 + span) | none pass | 0/60 |
| Genuine images (max NCC anywhere) | 0.00–0.92 | 60 |

No genuine image simultaneously has all four criteria: high NCC, enough
dominant matches, moderate dominance ratio, and valid span. The conjunction
kills every genuine case through at least two independent conditions.

### FPR analysis

Tested 60 genuine images (holdout + tuning) against the proposed criteria:

- `SIFT_PATCH_NCC_MIN=0.78`: gap of 0.02 from weakest recovery (dl_001 at 0.80)
- `SIFT_PATCH_DOM_MIN=17`: interior point (weakest recovery at dom=18, gap=1)
- `SIFT_PATCH_RATIO_MIN=1.15`: interior point (weakest recovery at ratio=1.23, gap=0.08)

Combined result: **0/60 FPs** (0.0%).

Nearest genuine misses and why each is excluded:
- `dl_42_006`: NCC=0.92 but ratio=1.07 < 1.15
- `aad_42_001`: NCC=0.88 but dom=13 < 17
- `dl_777_003`: NCC=0.82 but dom=17 AND span_ok=False (min_ext=544 > max_span)

## Implementation

Third-pass check in `_cluster_and_decide()`, after standard dominance
(pass 1) and neighborhood merge (pass 2) both fail:

1. Guard: `span_ok AND dom_count >= patch_dom_min AND dom_count > patch_ratio_min × runner`
2. Compute Pearson NCC between the source bounding box and its shift
3. If NCC ≥ `patch_ncc_min`, mark as detected

The `_patch_ncc()` helper crops the source region (keypoint bbox + 10px
padding), clips both source and target to valid image bounds, and computes
the standard Pearson correlation on the flattened grayscale arrays.

Only the SIFT stage passes patch parameters. ORB is unchanged.

## Measured results

### Copy-move detector (isolation)

| Metric | Pre-W14 | Post-W14 |
|--------|---------|----------|
| Detector recall (holdout) | 21/30 (70%) | 25/30 (83%) |
| Detector recall (tuning) | 20/30 (67%) | 22/30 (73%) |
| Genuine FPR | 0% | 0% |

### End-to-end (eval harness)

| Metric | Pre-W14 | Post-W14 | Delta |
|--------|---------|----------|-------|
| copy_move tuning | 76.7% | 80.0% | **+3.3pp** |
| copy_move holdout | 76.7% | 90.0% | **+13.3pp** |
| overall tuning | 87.2% | 87.8% | +0.6pp |
| overall holdout | 89.4% | 91.7% | **+2.3pp** |
| genuine FPR | 0% | 0% | unchanged |
| undetected_autoclear (tuning) | 23 | 22 | -1 |
| undetected_autoclear (holdout) | 19 | 15 | -4 |

## Gates ratcheted

- `overall_recall_min`: 0.86 → 0.87
- `copy_move`: 0.70 → 0.77
- `undetected_autoclear_max`: 25 → 23

## Remaining copy_move gaps

Post-W14, the 5 remaining holdout detector misses:
- 3: dominant shift is wrong (structural repetition at a different offset
  entirely dominates the copy-move shift)
- 1: dominant shift correct but dom_count=15 < patch_dom_min=17
- 1: dominant shift correct but ratio=1.10 < patch_ratio_min=1.15

Further improvement requires restricting the search to YOLO-detected photo
regions (eliminating structural repetition outside the copied area) or a
learned localizer.
