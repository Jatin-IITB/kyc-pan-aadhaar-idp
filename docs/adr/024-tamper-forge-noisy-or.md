# ADR-024: Tamper Forge & Noisy-OR Forensic Aggregation

**Date:** 2026-08-11
**Status:** Accepted
**Deciders:** Jatin Gupta

## Context

Phase 11 W2 built `tools/forge/tamper_forge.py` — a 6-class forgery generator
(copy-move, text-splice, font-swap, screen-recapture, EXIF-edit, regenerate),
each severity-parameterized and labeled with target region + expected detector.
Run against the existing forensics suite it produced the project's first honest
tamper-recall measurement: **26%**, versus the headline claim of 97%.

Root cause was not the individual detectors but the aggregator. `SpoofScorer`
combined signals as a **weighted average**. A forgery typically trips one
detector hard (a Photoshop EXIF tag scores 0.70), but `0.70 × 0.15 weight =
0.105`, below the 0.2 flag threshold — a single strong, correct signal was
averaged into silence.

## Decision

**1. Noisy-OR over gated per-detector scores.** Each detector contributes only
when its own decision gate fires (suspicious regions / detected / is_recaptured
/ software_edited / inconsistent regions); gated scores combine as
`1 - ∏(1 - sᵢ)`. One strong detector can now flag a document, weak signals
compound, and a genuine document (all gates closed) scores exactly 0.

**2. Font demoted to corroboration-only.** Stroke-width "consistency" has **no
separating power** on legitimately multi-font ID cards — measured genuine
0.22–0.71 vs attack 0.20–0.71, fully overlapping. Its prior is set below the
flag threshold (0.18) so it can reinforce but never flag alone. Real font
forensics needs OCR field-region context (deferred to W5).

**3. Screen detector rescored by peak prominence.** The old peak-density metric
maxed at 0.13 with genuine/attack overlap. Scoring by top-peak prominence over
the mid-band noise floor (`(peak − mean)/std`) separates a sharp screen-grid
Moiré from diffuse guilloche energy: at threshold 0.15, 0/30 genuine FP.

## Results (30 genuine + 180 forged synthetic docs)

| Attack | Recall | | Aggregate | |
|---|---|---|---|---|
| copy_move | 100% | | **Overall recall** | 26% → **44%** |
| exif_edit | 100% | | **Genuine FPR (synthetic)** | **0%** |
| screen_recapture | 67% | | **Real card (test_pan_v2)** | **PASS** |
| font_swap | 0% | | text_splice | 0% |
| regenerate | 0% | | | |

The 0% genuine false-positive rate is the load-bearing invariant: a KYC system
must never reject a real customer's document. The screen threshold is set
conservatively (0.25) so the one available real phone-captured card scores 0.000
and passes — real captures carry more mid-band energy than clean synthetic
renders, so screen recall is deliberately traded down until a real-capture
validation set exists. Tightening it on synthetic data alone would risk
false-flagging real customers.

## Consequences

**Positive:** honest, reproducible tamper metrics; the dilution bug that hid
correct signals is fixed; copy-move, EXIF, and screen recapture are reliably
caught at zero genuine false positives; `dominant_shift` and prominence values
are concrete HITL evidence.

**Negative & W4 backlog (now data-driven):** three attacks remain under-detected
and honesty requires stating so — `text_splice` needs block-wise ELA to localize
a recompression seam (global mean ELA cannot); `regenerate` (clean re-render)
needs no-capture-noise / PRNU analysis; `font_swap` needs OCR region context.
The 97% recall target is **not yet met** (currently 54% on the synthetic red
team) and remains aspirational pending W4.
