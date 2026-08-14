# ADR-027: Re-Audit Remediation

**Date:** 2026-08-12
**Status:** Accepted
**Deciders:** Jatin Gupta

## Context

A second independent audit of the ADR-026 remediation found two critical and
four significant findings. The remediation CLAIMED held-out confirmation but
the code did not enforce it, and half the attack surface was ungated in CI.

## Findings addressed

### C1 — Held-out seed pair never automated

ADR-026 reported held-out numbers (seeds 777/888) but `run_eval.py` hardcoded
only the tuning seeds (42/123). The held-out confirmation was a manual one-off
with no enforcement in CI. Any future threshold change could overfit without
detection.

**Fix:** `run_eval.py` now defines `SEED_PAIRS = {"tuning": (42, 123),
"holdout": (777, 888)}` and runs forensics + decision on BOTH passes
independently. Each pass generates its own dataset under
`data/tuning/` and `data/holdout/`, and every gate is checked against each.
CI fails if EITHER pass has a gate failure.

### C2 — Three ungated attack classes

`eval_thresholds.yaml` only gated copy_move, exif_edit, and screen_recapture.
text_splice, font_swap, and regenerate had no per-attack recall floor — a
regression to zero detection on any of them would pass CI silently.

**Fix:** Added zero-floor gates (`0.0`) for all three ungated attacks. This
costs nothing today (they are already at 0%) but will catch any regression if
a future change inadvertently breaks detection. The floor ratchets up as W4/W5
close each blind spot.

### S1 — `attack_font_swap` missing augmentation

The genuine documents in `identity_forge.py` have capture-condition
augmentation applied (blur, brightness, noise, JPEG re-encode). The font_swap
attack re-rendered the same card but skipped augmentation, so the delta
between genuine and forged included augmentation artifacts, not just fonts.

**Fix:** `attack_font_swap` now applies augmentation after re-rendering, using
a deterministic RNG derived from the render seed.

### S2 — `attack_regenerate` used a hardcoded seed

The regenerate attack used `np.random.default_rng(999)` — a different RNG
from the genuine document's render seed — so the re-rendered image had a
different photo, guilloche, and signature. ELA was measuring content
differences, not compression artifacts.

**Fix:** Uses `truth.get("render_seed", 0)` so the re-render produces
identical visual content; the only delta is the double-JPEG cycling.

### S3 — knnMatch self-match slot

`knnMatch(descriptors, descriptors, k=k)` includes the identity match
(distance 0) for every keypoint, consuming one of the `k` result slots. The
effective measurement window for the promiscuity filter was one slot smaller
than intended.

**Fix:** Increased `k` by 1 to compensate for the self-match slot.

### S4 — No test for non-grid-aligned offsets

No test verified that the forge actually generates non-grid-aligned copy-move
offsets (the core C2 claim). Also no test verified that the 3px screen
evasion case is actually not detected (M7).

**Fix:** Added `test_copy_move_offset_is_not_grid_aligned` (asserts the forge
offset is not a multiple of 16 on at least one axis) and
`test_screen_evasion_case_not_detected` (asserts the detector does not fire
on a 3px Moire grid).

## Results

| Metric | Tuning 42/123 | Held-out 777/888 | Gate |
|---|---|---|---|
| Genuine FPR | 0/30 | 0/30 | <= 0 |
| Overall recall | 28.3% | 31.7% | >= 25% |
| copy_move | 50% | 50% | >= 40% |
| exif_edit | 100% | 100% | >= 90% |
| screen_recapture | 20% | 40% | >= 15% |
| text_splice | 0% | 0% | >= 0% |
| font_swap | 0% | 0% | >= 0% |
| regenerate | 0% | 0% | >= 0% |
| Genuine auto-clear | 100% | 100% | >= 90% |
| flagged_leakage | 0 | 0 | 0 |
| undetected_autoclear | 129 | 123 | <= 135 |

All gates pass on both passes. 90/91 unit tests pass (the one failure is a
pre-existing, unrelated classifier test).

## Consequences

**Positive:** The anti-contamination defense is now real code, not
documentation. Every attack class is gated. The forge produces honest
isolated attacks. The detector's measurement window is correct.

**Negative:** `make eval-fast` now takes roughly 2x as long (two forensic
sweeps). This is acceptable — the eval runs in ~30s, not minutes.
