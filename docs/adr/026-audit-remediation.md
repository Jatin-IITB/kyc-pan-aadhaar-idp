# ADR-026: Truth Engine Audit Remediation

**Date:** 2026-08-12
**Status:** Accepted
**Deciders:** Jatin Gupta
**Supersedes numbers in:** ADR-022 (copy-move), ADR-024, ADR-025

## Context

An independent audit of Phase 11 (W1–W3) accepted the honesty architecture but
found the two most-quotable metrics — "0% genuine FPR" and "copy-move 100%
recall" — were materially weaker than stated, plus several eval-validity and
correctness issues. The headline problems:

- **C1 contamination:** every detector threshold was tuned on the exact
  30-genuine/180-forged set the metrics were reported on, with no held-out
  confirmation and no disclosure.
- **C2 attack–detector co-design:** the forge's copy-move offsets were always
  multiples of the detector's 16 px sampling stride, and screen Moiré periods
  were all inside the detector's frequency band. The audit empirically
  confirmed the v2 detector was blind at non-aligned offsets — so 100% was
  recall on the visible 1/16th.
- **C3 survivorship bias:** extraction F1 (97.9%) excluded 3/12 timed-out
  samples; counting them as misses drops it below its own 0.85 gate.
- **S2 confounded metric:** the font gate fired on nearly every genuine render,
  silently collapsing genuine auto-clear to 26.7% and propping up ~59 of the
  "detected" forgeries for the wrong reason.

## Decision

### Copy-move detector: grid-based → ORB (alignment-free)

Replace DCT block sampling with ORB keypoint self-matching. Keypoints anchor to
image content, not a sampling grid, so detection is offset-invariant. Guards
that survive the rewrite, plus new ones the harder red team forced:

- **Repetition filter** (promiscuous descriptors = texture) — retained.
- **Shift-vector clustering + dominance ratio** — one forgery is one
  displacement; periodic structure spreads across bins and fails dominance.
- **Region-shape band** — the source cluster must be compact and roughly
  square: `min_extent ≥ 32`, `max_extent ≤ 340`, `aspect ≤ 4`. Rejects thin
  repeated text lines, card-wide periodic structure, and monospace
  number/address strips.
- **Exact-copy match threshold** (`hamming ≤ 10`) — a true copy is a pixel
  copy (descriptors near-identical); repeated-but-different glyphs match only
  loosely. This is what finally separated a cloned photo from a repeated
  Aadhaar-number strip and drove genuine FPR to 0/60.

Low FAST threshold + 5000-keypoint budget so soft photo texture is not crowded
out by the card's strong text/QR corners.

### Forge de-gaming

- Copy-move duplicates the **whole photo region** at an **arbitrary,
  jittered, non-grid-aligned** offset (records the offset in the label).
- Screen `low` severity is a deliberate **evasion case** (3 px grid, above the
  detector's band) so the recall curve includes attacks that are undetectable
  by design instead of only detector-visible ones.
- `font_swap` re-renders the **same card** via a per-sample `render_seed` now
  recorded in truth, so the only delta from genuine is the fonts (previously it
  also changed photo/guilloche/signature, collapsing toward "clean re-render").

### Font de-scored (S2)

`SpoofScorer` font prior → 0. Stroke-width consistency has no separating power
on multi-font ID cards (genuine 0.22–0.71 vs attack 0.20–0.71, fully
overlapping). It is still computed and returned for reviewers, but contributes
nothing to the score. Result: genuine auto-clear returns to 100%, and the
blind-spot metric is no longer confounded.

### Eval harness

- **Held-out confirmation (C1):** a second seed pair (777/888) is generated and
  the same gates checked against it. Reported numbers are the held-out floor.
- **Honest extraction F1 (C3):** timed-out samples count as all-fields-FN;
  that number gates. F1-on-succeeded is reported alongside for context.
- **Dataset hard-fail (S3):** empty genuine or tamper set raises, never
  silently SKIPs to green.
- **Genuine auto-clear gate (S2):** new floor (0.90) catches any regression
  that tanks the auto-clear rate.
- Balanced round-robin extraction sampling across doc types.

### CI hygiene (S4, S5)

- Dependency versions pinned in `truth_engine.yaml` (thresholds are calibrated
  against specific render/CV behaviour).
- Legacy nightly `quality_gate.yaml` set to manual-dispatch only (it references
  committed data dirs that do not exist and can never pass).
- ADR-024 "54%" and ADR-025 "spoof ≥ 0.15" corrected.

## Results (honest, held-out confirmed)

| Metric | Tuning 42/123 | Held-out 777/888 | Gate |
|---|---|---|---|
| Genuine FPR | 0/30 | 0/30 | ≤ 0 ✓ |
| Overall tamper recall | 28.9% | 31.7% | ≥ 25% ✓ |
| copy_move | 46.7% | 50.0% | ≥ 40% ✓ |
| exif_edit | 100% | 100% | ≥ 90% ✓ |
| screen_recapture | 23.3% | 40.0% | ≥ 15% ✓ |
| Genuine auto-clear | 100% | 100% | ≥ 90% ✓ |
| flagged_leakage | 0 | 0 | 0 ✓ |
| undetected_autoclear | 128 | 123 | ≤ 135 ✓ |

Tuning and held-out track within a few points on every metric — the evidence
generalizes. The headline numbers are now lower than the pre-audit figures
(copy-move 100%→~50%, overall 44%→~30%) **because they are honest**; that
downward correction is the entire purpose of having built the Truth Engine.

## Consequences

**Positive:** metrics survive a held-out check; the copy-move detector catches
arbitrary-offset duplication (the real threat); genuine auto-clear is healthy;
the W4 backlog (text_splice, regenerate, font_swap, screen recall) is now an
honest, data-driven target instead of a rigged one.

**Negative:** ORB self-matching is heavier than grid DCT (still <150 ms/doc);
copy-move recall (~50%) reflects genuine difficulty of detecting a duplicated
region among ~30% of arbitrary offsets that yield too few stable keypoints —
further gains need multi-scale or dense-field matching (future W4). The single-
line text-splice surface is explicitly out of scope for the copy-move detector.
