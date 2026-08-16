# ADR-028 — W4: Forensic Precision Pass

**Status:** Accepted  
**Date:** 2026-08-16  
**Decision Makers:** Jatin Gupta

## Context

After W1-W3 and the re-audit remediation, `text_splice` and `regenerate` attack
classes had 0% recall. The root cause was twofold:

1. **Forge normalization erasure** — `forge_dataset()` re-saved all forged images
   at Q=92, normalizing compression artifacts. A text splice at Q=78 lost its
   quality seam when re-saved at Q=92.

2. **Detector gap** — ELA's fixed-threshold contours detect gross pixel edits
   but not subtle quality differences. No detector could distinguish a Q=78
   JPEG from a Q=92 JPEG by pixel analysis alone.

## Approaches Tried and Rejected

### Block-wise ELA (MAD z-score outlier detection)
Divided the image into 16×16 blocks, computed per-block mean ELA residual, and
flagged blocks whose z-score exceeded 4.0 (MAD-based). **Result:** 77-87%
genuine FPR. Natural texture variation in genuine ID cards produced enough
outlier blocks to trigger detection on nearly every genuine image.

### JPEG quality estimation via residual minimization
Re-compressed the image at Q=40-95, measured residual vs the original, and picked
the Q with minimum residual. **Result:** On synthetic images the residual curve
is monotonically decreasing (minimum always at Q=95 regardless of actual save
quality). The method assumes the image was JPEG-compressed *from a natural photo*,
but our synthetic renders have unnaturally uniform DCT distributions.

## Decision

### D1: Fix the forge to preserve attack quality
`attack_text_splice` and `attack_regenerate` now return a 3-tuple
`(image_array, label, raw_jpeg_bytes)`. The raw bytes are JPEG-encoded at the
attack quality (text_splice: Q=70-85 by severity; regenerate: Q=45-75).
`forge_dataset()` writes these bytes directly instead of re-encoding via
`cv2.imwrite`, preserving the quantization tables.

### D2: JPEG quality estimation via quantization tables
Read the JPEG luminance quantization table directly from the file (via
`PIL.Image.quantization`), reverse the IJG scaling formula to estimate save
quality. This is an exact mathematical inverse — no statistical estimation,
no signal-processing heuristics.

Threshold: `jpeg_quality < 88` flags as suspicious.

### D3: Wire into MetadataForensics, not ELA
Quality estimation is a file-format property (quantization tables live in the
JPEG header), not a pixel analysis. It belongs in `MetadataForensics.analyze()`
which already handles file-level properties (EXIF, software tags).

### D4: Clean up dead code
- Removed `_block_ela()` method from `ELADetector` (rejected approach)
- Removed dead `low_quality_jpeg` evidence path from `SpoofScorer`
- Fixed `test_forensics.py` tests to match the new architecture

## Honest Limitations (Re-audit, 2026-08-16)

### L1: Tautological detection — catches forge signature, not tampering
The detector achieves 100% recall on our forge because it detects "saved at
Q ≠ 92", not "pixels were tampered with." A real attacker who edits a document
and saves at Q=92 (one setting in any image editor) bypasses the detector
completely. ELA also does not fire on text_splice at any quality level — the
entire detection for these two attack classes rests on a single metadata feature.

### L2: Eval circularity
The forge saves at Q=70-85, the detector fires at Q<88, the eval tests on
forge output. This is a closed loop. Changing the forge's save quality to Q=92
drops recall to 0% instantly, proving the metric measures forge-detector
coupling, not detection capability against an adversary.

### L3: Real-world FPR unknown
The 0% genuine FPR is valid only against Identity Forge output (Q=92). Real
KYC submissions arrive via phone cameras (Q=75-95), WhatsApp (Q=60-75),
scanners (Q=70-90), and email (Q=65-80). The Q<88 threshold would false-
positive on a large fraction of legitimate submissions. Production deployment
requires calibrating against real scan quality distributions.

### L4: No defense in depth
`text_splice` and `regenerate` detection relies on ONE feature (JPEG quality).
There is no pixel-level, frequency-domain, or semantic backup detector for
these attack classes. Robust detection requires approaches that analyze content
(double-JPEG ghost detection, DCT coefficient histogram analysis, or
learned tampering classifiers).

## What the quality detector IS good for

Despite the limitations, the detector provides real value:

1. **Catches lazy/opportunistic forgeries** — most real-world document fraud
   involves screenshots, WhatsApp-forwarded images, or cheap editing apps
   that save at non-standard quality. These are the majority of attacks.

2. **Zero-cost corroboration signal** — reading a quantization table is O(1).
   Combined with other metadata signals (EXIF software tags, date anomalies),
   it strengthens the metadata gate without adding latency.

3. **Honest forge improvement** — the forge now produces more realistic attack
   artifacts (actual compression differences vs. normalized-away artifacts),
   which makes ALL detectors' eval numbers more honest even if this specific
   detector is limited.

## Measured Results (synthetic eval, NOT production-representative)

| Metric | Before W4 | After W4 | Note |
|---|---|---|---|
| genuine FPR | 76.7% | 0.0% | Synthetic only — real-world unknown |
| text_splice recall | 0% | 100% | On our forge only — trivially evadable |
| regenerate recall | 0% | 100% | On our forge only — trivially evadable |
| overall recall | 25% | 61.7% | |
| genuine auto-clear | 23.3% | 100% | Synthetic only |

Gate ratchets are set conservatively (0.90 for text_splice/regenerate) to catch
regressions, not to claim adversarial robustness.

## Open Work (beyond W4)

Robust text_splice and regenerate detection requires pixel-level or frequency-
domain approaches that survive quality normalization:
- Double-JPEG ghost detection (aligned vs misaligned block grids)
- DCT coefficient histogram analysis (first-digit distribution anomalies)
- Learned tampering localization (e.g., ManTraNet, CAT-Net)
- PRNU sensor noise analysis for regenerate

These are research-grade problems. The quality detector is a pragmatic first
step, not a solution.
