# ADR-040: W16 — The text_splice/regenerate tautology, measured; and why double-JPEG detection does not replace it

**Status:** Accepted (measurement + negative result; no detector shipped)
**Date:** 2026-09-04
**Extends:** ADR-028 (JPEG-quality forensics and its caveat)
**Related:** ADR-031, ADR-033 (prior negative results)

## Context

ADR-028 flagged `text_splice` and `regenerate` recall as tautological: both
are caught by the `low_jpeg_quality` metadata gate, which fires below
`MetadataForensics.LOW_QUALITY_THRESHOLD = 88`. The tamper forge saves
`text_splice` at Q=85/78/70 and `regenerate` at Q=75/60/45 — all beneath the
gate — while genuine documents are written at Q=92. The detector reads save
quality, not tampering.

The README has carried this as an *assertion* ("trivially evadable by a
competent adversary saving at Q>=88"). W16 turns it into a measurement, then
asks whether a pixel/frequency-domain detector can replace the gate.

## 1. The tautology, measured

`tools/eval/evasion_probe.py` re-runs both attacks across a sweep of attacker
save qualities, intercepting `cv2.imencode` inside `tamper_forge` so that the
painted splice, the re-render, and the double-compression loop are otherwise
byte-identical to a normal forge run. Only the save quality varies.

Holdout split, 3 doc types x 3 severities x 10 documents = 90 per attack:

| Attack | recall at forge quality | recall at Q=95 |
|---|---|---|
| `text_splice` | 100% | **7.8%** |
| `regenerate` | 100% | **10.0%** |

At Q=95 the `low_jpeg_quality` flag fires 0/180 times and the mean spoof score
is 0.045–0.063 against a 0.2 flag threshold. A one-line change in the
attacker's save quality removes both detections. The caveat was accurate and
is now quantified.

## 2. Textbook double-JPEG detection is INVERTED on this corpus

The obvious replacement is double-quantization analysis. It fails here, and
fails in the dangerous direction. The compression histories are:

```
genuine        render -> augment(JPEG q in [70,95)) -> decode -> JPEG@92
regenerate@Q   render -> [JPEG@Q -> decode] x2 -> JPEG@Q
text_splice@Q  genuine -> paint region -> JPEG@Q
```

Genuine documents are **mismatched double-JPEG** (q1 != q2) — the classic
*detectable* case. `regenerate` is compressed three times at a **single**
quality — the classic *near-idempotent, undetectable* case. Measured DQ
periodicity (mean over five low-frequency AC coefficients):

| population | dq_mean | blockiness |
|---|---|---|
| genuine | 2.584 | 1.153 |
| `regenerate@95` | **1.976** | 0.940 |
| `text_splice@95` | 2.587 | 1.153 |

`regenerate` carries *less* double-quantization signal than genuine
(delta −0.608). A detector keyed on "more DQ => forged" fires on genuine
documents and misses the attack. `text_splice` is statistically identical to
genuine on global features (delta +0.002) because it alters one small
rectangle and whole-image statistics are dominated by the untouched card.

### A rejected second tautology

`blockiness` appears to separate `regenerate` from genuine (genuine p10 =
1.001 vs attack p90 = 0.989). It was **rejected, not shipped**. The gap is
produced by the forge not applying augmentation to `regenerate` output, so a
detector built on it would key on a pipeline artifact — replacing one
tautology with another. This is the failure mode ADR-028 exists to prevent.

## 3. Local JPEG-ghost: signal exists (oracle), blind search fails

`text_splice` does contain real tampering, so W16 tested local JPEG-ghost
localization: recompress at a probe quality and compare the residual inside
the painted region against the rest.

Image-level ghost statistics were discarded up front — recompressing a Q=92
genuine at Q=92 is near-idempotent (residual ~ 0) while anything saved at
Q=95 has a large residual everywhere, so any image-level statistic re-reads
save quality. The only honest test is within-image.

**Oracle** (handed the ground-truth box, therefore an upper bound on any blind
detector), in-box / out-of-box residual ratio:

| probe Q | splice | genuine control | delta |
|---|---|---|---|
| 88 | 4.801 | 2.327 | +2.474 |
| **92** | **5.762** | **0.966** | **+4.796** |
| 95 | 0.660 | 2.214 | −1.554 |

At probe Q=92 — the host's original save quality — splice p10 (2.906) exceeds
genuine-control p90 (2.246). The signal is real. It inverts at Q=95, which
confirms the mechanism rather than a coincidence: probing at the host's
quality drives the host residual down while the painted region, which never
saw Q=92, stays high.

**Blind** (no box; sweep probe qualities, scan candidate windows, take the
max) with a ratio statistic **fails and inverts**:

| population | mean | p90 | max |
|---|---|---|---|
| genuine | 14.987 | 22.404 | 22.946 |
| `text_splice@95` | 7.871 | 11.991 | 16.959 |

No usable operating point. The cause is diagnostic, not incidental: all 30
genuine images selected probe q=92 — their own save quality — where the ghost
map is near-zero everywhere, so a ratio of means explodes on noise. The
oracle avoided this by averaging over one large fixed region.

The indicated fix is a difference statistic (residual units, cannot blow up on
a near-idempotent map), implemented as `--stat diff` in
`tools/eval/double_jpeg_probe.py`. **It has not been measured.** The
evaluation corpus was deleted mid-investigation and regeneration was
unavailable, so this ADR records the approach as open rather than claiming a
result. Every number above was measured before that point and is reproducible
with the committed tools after `make forge`.

## Decision

No detector is shipped. `text_splice` and `regenerate` gates remain
metadata-only and remain tautological, now with a measured evasion curve
rather than an assertion.

Specifically:

1. Textbook double-JPEG detection is rejected for this corpus, with the
   inversion documented so it is not re-attempted.
2. Global blockiness is rejected as a second tautology.
3. Local JPEG-ghost is **open**: oracle-validated, blind search unresolved.

## Consequences

- The README's `\*` footnote on `text_splice` / `regenerate` should cite the
  measured collapse (100% -> 7.8% / 10.0% at Q=95) instead of asserting
  evadability.
- `per_attack_recall_min` for both attacks stays at 0.90. Those gates catch
  regressions in the metadata path; they are not robustness claims. This was
  already documented in ADR-028 and remains true.
- Future work on this pair should start from `--mode blind --stat diff`, and
  must clear 0% genuine FPR on >= 120 genuine documents before it is
  considered, per the ADR-031 precedent where a promising retune measured 2.5%
  FPR at n=120 after looking clean at n=30.

## Reproduction

```bash
make forge
.venv/bin/python -m tools.eval.evasion_probe                      # section 1
.venv/bin/python -m tools.eval.double_jpeg_probe --mode dq        # section 2
.venv/bin/python -m tools.eval.double_jpeg_probe --mode oracle    # section 3
.venv/bin/python -m tools.eval.double_jpeg_probe --mode blind --stat diff
```
