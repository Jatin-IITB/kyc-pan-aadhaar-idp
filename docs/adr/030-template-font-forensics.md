# ADR-030 — W6: Template-Conformance Font Forensics

**Status:** Accepted
**Date:** 2026-08-17
**Decision Makers:** Jatin Gupta

## Context

`font_swap` sat at exactly **0.000** recall across every prior eval. Not low —
zero. That precision is itself the diagnostic: the detector was not mistuned,
it was structurally incapable of seeing the attack.

`FontConsistencyAnalyzer` flags regions whose stroke width deviates from the
document's own median. But `attack_font_swap` re-renders the *entire card*
through `font_override`, permuting font roles globally. Every value line shifts
together, so the document remains perfectly self-consistent. There is no
outlier to find. ADR-026 had already set the font prior to 0 after intra-document
metrics showed fully overlapping ranges on genuine vs attacked cards.

## Decision

### D1: Ask template conformance, not internal consistency

Genuine cards carry *designed* typographic structure — `templates.py` renders
labels in `regular`, values in `bold`, and ID numbers in `mono`. The attack
permutes exactly those roles. So the detectable invariant is not "is this
document self-consistent?" but "does its typography match the issuing template?"

`services/forensics/font_profile.py` extracts a three-feature signature and
compares it against a per-doc-type envelope calibrated from genuine documents.

| Feature | Direction under swap | Mechanism |
|---|---|---|
| `corner_top` | rises | serif terminals add corners that sans lacks |
| `mod_top` | falls | serif modulates thick/thin; bold sans is near-uniform |
| `adv_cv_min` | rises | monospace ID numbers become proportional |

Directions are fixed from mechanism, not fit, so a lucky sample cannot flip them.

### D2: Exclude the masthead band

Templates legitimately set headers in a different face from body text —
Aadhaar renders "Government of India" in `serif` by design. Including that band
widens the genuine envelope enough to swallow the swap signal. The top 18% is
dropped before feature extraction.

### D3: Selection must estimate out-of-sample FPR, and did not at first

The genuine-FPR-zero invariant is non-negotiable, and three successive
selection rules each looked correct and each leaked. Measured, in order:

| Selection rule | Own-holdout FPR | Eval-set FPR |
|---|---|---|
| in-sample calibration FPR = 0 | 8.3% | — |
| leave-one-out FPR = 0 | 2.1% | — |
| LOO = 0, stable over 1 grid step | 0.0% | 3.3% |
| **LOO = 0, stable over 2 grid steps** | **0.0%** | **0.0%** |

The lesson generalizes past this detector: *the first threshold that clears a
constraint sits on that constraint's boundary and will not survive new data.*
Within any given rule the margin is chosen without consulting the holdout
split. One honest qualification (audit S2): the *rule itself* was iterated
while observing holdout outcomes — the table above is that history — so the
final figure is out-of-sample with respect to margin selection but not with
respect to rule selection. The eval harness's independent seeds provide the
cleaner check, and the harness now hides holdout columns during sweeps unless
`--reveal-holdout` is passed. The final config (margin 8.0, vote 1) is an
interior point by a real distance.

### D4: A separate prior, not a revival of the old one

`font_template` enters `SpoofScorer` as its own signal at prior **0.80**. The
legacy `font` prior stays at 0 — ADR-026's finding about stroke-width
consistency still holds; this is a different measurement that earned a prior by
independent validation.

## Measured Results (post-audit, reproducible profile)

Calibration on seed 9001, own-holdout on seed 24601, then validated against the
eval harness's independent genuine/tamper sets (seeds 42/123 and 777/888).

| Metric | Before | Tuning | Holdout |
|---|---|---|---|
| `font_swap` recall | 0.000 | **0.467** | **0.467** |
| overall tamper recall | 0.639 | **0.733** | **0.756** |
| genuine FPR | 0.000 | **0.000** | **0.000** |
| undetected auto-clear | 69 | **48** | **44** |

Detector cost: p50 ~4.6 ms, p95 ~6.8 ms.

Per doc type on the calibration holdout (recall at 0% FPR): PAN 81.2%,
driving_license 75.0%, **Aadhaar 6.2%**. Recorded in
`config/font_profiles.json` under `measured_holdout`.

## Independent Audit (2026-08-17)

The audit's adversarial checks confirmed the feature math, the LOO
implementation, the grid-edge stability logic, the noisy-OR monotonicity, and
the gate arithmetic — and found one critical defect:

**C1 — the first emitted profile was not reproducible.** `build_data`'s cache
check looked only at `cal/genuine/pan`, so the envelope had been fit on stale
cached data that the committed seeds could not regenerate. The recorded holdout
numbers were honest *for the shipped bounds* (independently re-verified:
recall 0.604, FPR 0.0 on 96 fresh genuine docs), but a routine re-emit would
have silently replaced the profile with a different one.

Resolution: cache validity now requires every doc type at the exact requested
count; the emitted JSON carries a content hash of all calibration inputs, the
extractor version (checked at load — a mismatched profile is refused), and n.
The profile was re-emitted from forced-clean data and **round-trips
byte-identically**. The honest cost: holdout recall 0.604 → **0.542**
(driving_license 93.8% → 75.0% — the stale data had flattered it), and eval
font_swap 0.533 → 0.467. The numbers above are the reproducible ones.

**S1 — fail-open crash paths.** Grayscale input crashed `signature()` and a
malformed profile spec crashed `analyze()`; both landed in the forensics node's
blanket except, which returns `spoof_score 0 / PASS` — discarding the *other*
detectors' results too. Fixed: 2-D/BGRA inputs handled, malformed specs skipped
with a warning, and the template-font call isolated in its own try so its
failure degrades to "signal absent" rather than zeroing the stack. Regression
tests added for each path.

**S2 — holdout consumption at the rule level.** The margin is chosen blind, but
the selection *rule* itself was iterated four times while watching holdout
numbers (the D3 table is that history). The final 0% is therefore not a fully
clean out-of-sample estimate at the rule level. The harness now hides holdout
columns unless `--reveal-holdout` is passed; D3's claim is qualified
accordingly.

**S3 — no extractor/profile version binding.** Fixed as part of C1's
resolution.

## Limitations

### L1: Aadhaar is effectively undetected
6.2% recall. PAN and DL both clear 80%. The Aadhaar layout is number-dominant
and its genuine envelope is wide enough that the swap stays inside it. Excluding
the masthead helped the other two types but not this one. Aadhaar needs its own
feature — likely operating on the ID-number line in isolation rather than on
ink-ranked lines.

### L2: Calibrated against our own renderer
Envelopes are fit on Identity Forge output. Real PAN/Aadhaar cards vary by
issue year, print vendor, and wear. Production use requires recalibration
against genuine scans, and the envelope would almost certainly need to be wider —
which would cost recall.

### L3: Bounded by the attack's own construction
The forge swaps between four bundled font roles. An adversary matching the
issuing font exactly produces no signature to detect. This raises the cost of a
forgery; it does not make one impossible.

### L4: Small calibration sets
16 genuine documents per type per split. The D3 table shows how sharply
conclusions moved as the selection rule tightened — a larger calibration set
would allow a tighter margin, and therefore more recall, at the same FPR.

## Consequences

Gates ratcheted: `overall_recall_min` 0.55 → 0.70, `font_swap` 0.0 → 0.40,
`copy_move` 0.40 → 0.50, `screen_recapture` 0.15 → 0.30,
`undetected_autoclear_max` 75 → 50. The font_swap floor sits one sample below
its measured 0.467, not at it — a gate on the measurement boundary flips red
on single-sample noise (the D3 rule applied to gates themselves).

`copy_move` (0.63, ADR-031) and Aadhaar font conformance (L1) remain the
largest gaps, then `screen_recapture` (0.43).
