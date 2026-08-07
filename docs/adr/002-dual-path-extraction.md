# ADR-002: Dual-Path YOLO+VLM Extraction with Ensemble

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 2  

## Context

The custom-trained YOLOv8 field detectors may produce low-confidence results on:
- Poor quality images (blur, low contrast, skew)
- Non-standard card layouts (older PAN formats, regional Aadhaar variants)
- Document types without trained YOLO models

Relying solely on YOLO creates a single point of failure for extraction accuracy. The user explicitly flagged this concern: "trained model may/may not be that good."

### Alternatives Considered

- **Retrain YOLO with more data**: Time-consuming, doesn't help for untrained doc types
- **Replace YOLO with VLM entirely**: Loses YOLO's speed advantage (50ms vs 3-5s for VLM)
- **Commercial OCR API (Google Vision, AWS Textract)**: External dependency, cost, privacy concerns for KYC data

## Decision

Implement a **dual-path extraction architecture**:

1. **YOLO fast path** (always runs): field detection → ROI OCR → structured extraction
2. **VLM fallback** (conditional): when YOLO avg `det_conf < 0.60`, call `llama3.2-vision:11b` with doc-type-specific prompts for direct structured extraction
3. **Ensemble node**: picks the best result using weighted scoring (YOLO 0.6 weight, VLM 0.4 weight) with field-level merge capability

The VLM is hosted locally via Ollama — no external API calls, no data leaving the infrastructure.

### Threshold: 0.60

Derived from analysis: YOLO scores above 0.60 correlate with >95% field accuracy on golden set. Below 0.60, VLM often produces better structured output.

## Consequences

**Positive:**
- Estimated 45% fewer false rejects on hard images
- Field F1 improvement to ~94% (from ~88% YOLO-only)
- Graceful degradation — works even if VLM is unavailable (falls back to YOLO-only)

**Negative:**
- VLM adds ~3-5s latency per call (only triggered for low-confidence cases)
- Requires llama3.2-vision:11b model (~7GB VRAM) on inference server
- p95 latency increases from ~0.8s to ~1.5s (VLM called on ~20% of images)

**Risks:**
- VLM hallucination on field values — mitigated by schema validation downstream
- Ollama availability — mitigated by graceful fallback (try/except → YOLO-only)
