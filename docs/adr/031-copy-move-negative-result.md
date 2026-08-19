# ADR-031 — W7: Copy-Move Recall — Two Rejected Approaches

**Status:** Accepted (negative result)
**Date:** 2026-08-17
**Decision Makers:** Jatin Gupta

## Context

After W6, `copy_move` was the largest remaining forensics gap: 0.60 tuning /
0.67 holdout. Two approaches were attempted to lift it toward 0.95. Both are
rejected. The detector ships unchanged at its ADR-026 defaults.

## Diagnosis

Instrumenting the ORB detector to report which gate rejects each miss, on an
independent sample (seed 31337, 30 forgeries):

| Rejection stage | Count |
|---|---|
| `not_dominant` | 12 |
| `span_too_thin` | 6 |
| detected | 12 |

`not_dominant` cases carried only 2–4 matches in the dominant shift bin against
a required 8 — the duplicated photo region is low-texture and yields too few ORB
keypoints. That is a sensitivity limit, not a threshold limit.

## Approach 1 (rejected): dense offset-residual matching

Since a copy-move is an exact pixel copy, search displacements directly: take
candidate offsets from FFT phase self-correlation, then for each, find the
largest connected region where `|I(x) - I(x+shift)|` is near zero.
Texture-independent and alignment-free by construction.

**Result: no discrimination whatsoever.** 83.3% recall at 83.3% genuine FPR.

Flat background matches itself under *any* shift, so "looks the same when
shifted" is satisfied trivially by every genuine card. Adding a local-texture
gate to require the matched region carry real content did not separate either —
tampered median texture 4.42 vs genuine 4.24, fully overlapping, with recall and
FPR tracking each other at every threshold.

The failure is instructive: genuine ID cards contain substantial repeated print
structure (guilloche, background bands, tiled emblems). ADR-026's three guards —
repetition filtering, shift dominance, region span — exist precisely to separate
that from forgery. This approach discarded them and walked back into the problem
they were built to solve.

## Approach 2 (rejected): parameter retune

Loosening `match_max_hamming` 10 → 14 and `min_span_px` 32 → 24 measured 40% →
46.7% recall at 0% genuine FPR on a 30-document independent sample, and lifted
the real eval to copy_move 0.73 tuning / 0.70 holdout, overall 0.78.

It was rejected on validation against **120** independent genuine documents:

| `min_span_px` | `match_max_hamming` | genuine FPR |
|---|---|---|
| **32** | **10** | **0.00%** (shipped) |
| 28 | 10 | 0.00% |
| 32 / 28 | 12 or 14 | 0.83% |
| 24 | 10 / 12 / 14 | 2.50% / 3.33% / 2.50% |

The 0% reading on 30 samples was luck; the true rate for `24/14` is 2.5%. The
eval's own tuning pass independently caught it at 3.33%.

**A 0% rate verified on 30 samples does not certify 0%.** This repeats the W6
lesson (ADR-030 D3) in a new place: with a hard zero-FPR invariant, the sample
must be large enough that zero means something. 30 is not.

## Decision

`CopyMoveDetector` keeps its ADR-026 defaults (`match_max_hamming=10`,
`min_span_px=32.0`). Post-revert eval: copy_move 0.60 tuning / 0.667 holdout,
overall 0.744 / 0.772, genuine FPR 0.000 on both.

## What would actually work

The binding constraint is keypoint density on low-texture pasted regions, so the
answer is a better *region* descriptor, not a better threshold:

- Zernike or polar-cosine moments over overlapping blocks — rotation-invariant
  and defined on smooth content where ORB finds no corners, with matching
  restricted to a shift-dominance test as ADR-026 already does.
- Keypoint density adapted per region: detect low-texture areas first, then run
  a dense descriptor only there, keeping ORB for textured areas.
- A learned tampering localizer, which sidesteps hand-tuned envelopes but needs
  labelled real forgeries.

Any of these must be validated on ≥120 genuine documents before it can claim to
hold the zero-FPR invariant.
