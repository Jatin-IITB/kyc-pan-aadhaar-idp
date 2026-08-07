# ADR-004: Pure OpenCV/Numpy Document Forensics

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 4  

## Context

KYC processes must detect tampered, spoofed, or re-photographed documents. Options ranged from commercial fraud detection APIs to open-source signal analysis.

### Alternatives Considered

- **Commercial APIs (Jumio, Onfido, Veriff)**: High cost ($0.50-2.00/check), external data transmission (KYC privacy concern), vendor lock-in
- **Deep learning forgery detection (FaceForensics++, etc.)**: Requires training data of forged KYC documents (scarce), large model footprint
- **Manual review only**: Not scalable, high human cost

## Decision

Implement a **5-signal forensics suite** using only OpenCV and numpy — no external APIs, no ML model training required:

| Signal | Technique | Weight |
|--------|-----------|--------|
| ELA | JPEG re-compression diff | 0.25 |
| Copy-Move | DCT block matching | 0.25 |
| Font Consistency | Morphological stroke/height analysis | 0.15 |
| Metadata | EXIF editing software detection | 0.15 |
| Screen Recapture | FFT Moire pattern detection | 0.20 |

`SpoofScorer` aggregates into a single `spoof_score` (0.0-1.0) with risk levels: LOW (<0.2), MEDIUM (0.2-0.4), HIGH (0.4-0.7), CRITICAL (>0.7).

Forensics runs as a **parallel branch** in the graph (alongside extraction) — zero additional latency on the critical path.

## Consequences

**Positive:**
- Zero external API cost, zero data privacy risk
- Runs in parallel with extraction (<200ms for all 5 signals)
- Each signal is independently testable and tunable
- No ML training data required

**Negative:**
- Signal-based forensics has higher false positive rate than learned models
- ELA is JPEG-specific — may not work on PNG inputs (mitigated: we re-encode)
- No face liveness detection (out of scope for document-level forensics)
- Thresholds (0.4 review, 0.7 reject) may need tuning per deployment

**Risks:**
- Sophisticated forgeries may evade signal-based detection — acceptable for first version; can layer ML-based detection later
