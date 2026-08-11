# ADR-023: Identity Forge — Synthetic Document Generation

**Date:** 2026-08-11
**Status:** Accepted
**Deciders:** Jatin Gupta

## Context

Phase 11 (ADR-021) requires a labeled dataset to certify extraction F1, forensic
FPR/recall, and decision metrics. Real KYC documents cannot be collected or stored
(PII, legal exposure), and public Indian-ID datasets are scarce and unlabeled.

## Decision

Build `tools/forge/` — a deterministic synthetic document generator:

- **Identities:** Faker(`en_IN`) people; father's name shares the surname; all
  values fabricated.
- **Document numbers are structure-valid, not random strings:** PAN follows the
  official grammar (4th char entity code, 5th char surname initial); Aadhaar
  numbers carry a correct **Verhoeff check digit** (UIDAI's actual algorithm,
  implemented and unit-tested against reference vectors). This exercises the same
  validation paths real documents take.
- **Templates are structurally faithful, deliberately non-counterfeit:** correct
  layout, labels, guilloche-style security patterns, synthetic photos, real QR
  codes — but placeholder emblems and no replicated security features. These are
  eval artifacts, not forgeries of real documents.
- **Every sample ships three artifacts:** rendered JPEG, `truth.json` (field
  values + pixel bboxes + augmentations), YOLO-format labels. The bboxes fall out
  of the renderer for free, giving Phase 15 YOLO training data at zero labeling
  cost.
- **Augmentation policy (v1):** photometric ops (blur, brightness, shadow, noise,
  JPEG) preserve boxes exactly; 90° rotations transform them; small-angle rotation
  and perspective are deferred to a boxes-unavailable hard-eval split.
- **Determinism:** one `--seed` reproduces the entire dataset bit-for-bit.

## Validation performed

- 30 genuine synthetic docs (PAN/Aadhaar/DL): **0 forensic false positives**
  across copy-move, ELA, and screen-recapture — including dense guilloche
  backgrounds, the exact pattern class behind the ADR-022 incident.
- Planted photo-duplication attack on a synthetic card: detected at 1.0
  confidence with correct displacement evidence.
- VLM round-trip on a synthetic PAN: **4/4 fields exact-match** against ground
  truth — the forge → extract → score loop works end-to-end.

## Consequences

**Positive:** unlimited labeled data, no PII, reproducible benchmarks, free
detector training labels, tamper attacks get precise region ground truth (W2).

**Negative:** synthetic renders are cleaner than phone captures — metrics carry an
"on synthetic benchmark" qualifier until a small real-capture validation set
exists; Hindi/Devanagari text is out of scope for v1, so bilingual layouts are
not represented.
