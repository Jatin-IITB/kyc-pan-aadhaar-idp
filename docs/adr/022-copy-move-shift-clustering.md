# ADR-022: Copy-Move Detection via Shift-Vector Clustering

**Date:** 2026-08-11
**Status:** Accepted
**Deciders:** Jatin Gupta

## Context

The DCT block-matching copy-move detector scored 1.0 ("20 matched pairs") on genuine
PAN cards. Diagnosis on a real card: **1,914 block pairs with cosine similarity
> 0.99999** — printed security patterns (guilloche lines, tiled emblems, watermarks)
produce pixel-identical blocks *by design*. Threshold tuning (0.98 → 0.9997) and
variance filtering could not help: the false matches are perfect matches. The naive
"count similar pairs" statistic cannot distinguish repeating print structure from
forgery.

## Decision

Rewrite the detector around how forgery actually differs from print structure:

1. **DC removal** — zero the DC coefficient in the DCT feature so shared brightness
   can never drive similarity; only structural (AC) agreement counts.
2. **Repetition filter** — a block matching more than `max_neighbor_matches` (2)
   distant blocks belongs to a repeating texture and is discarded entirely. A genuine
   copy-move source/destination matches exactly once; a guilloche block matches
   hundreds of siblings.
3. **Shift-vector clustering** — a duplicated region moves by a single offset, so
   real forgery concentrates its matched pairs in one quantized displacement bin
   (8 px). Detection requires `min_matches` pairs **in the dominant bin**, not
   scattered anywhere in the image. (Fridrich-style shift histograms.)

Larger blocks (32 px) and 6×6 DCT features raise per-block distinctiveness.

## Consequences

**Positive:** genuine documents with dense repeating print no longer false-positive;
detection now keys on the one property forgeries cannot avoid (a coherent
displacement); `dominant_shift` is reportable as concrete evidence in the HITL UI.

**Negative:** duplications that are also periodic in the original design (an emblem
printed twice by the template) could still alias; pasted regions shifted by non-grid
offsets lose block alignment — recall vs. offset subtlety will be measured and tuned
by the Phase 11 tamper forge rather than by hand.
