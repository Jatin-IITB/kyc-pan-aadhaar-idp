# ADR-003: VLM-Only Extraction for Non-YOLO Document Types

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 3  

## Context

The project needed to expand from 2 document types (PAN, Aadhaar) to 12+ types (Passport, Driving License, Voter ID, Utility Bill, Bank Statement, Visa, Employment Letter, Tax Return, Insurance, Birth Certificate, Marriage Certificate). Training YOLO models for each new type would require:

- Thousands of labeled training images per type
- Weeks of annotation and training time
- Ongoing maintenance of 12+ separate models

### Alternatives Considered

- **Train YOLO for all types**: Impractical timeline, annotation cost, maintenance burden
- **Template matching**: Brittle, breaks on layout variations
- **Regex-only on raw OCR**: No spatial understanding, low accuracy on multi-field documents

## Decision

For document types without YOLO models, **skip YOLO entirely** (set confidence=0) and route to VLM extraction. The VLM receives the document image with a doc-type-specific prompt that lists expected fields and their formats.

Structure is enforced by:
1. **Per-type VLM prompts** with explicit field lists and format instructions
2. **JSON Schema validation** (Draft-07) for each doc type — 13 schema files
3. **Extended normalizers** — type-specific field normalizers (passport number, DL number, IFSC, etc.)

## Consequences

**Positive:**
- Instant support for 12+ doc types without any YOLO training
- Single VLM model handles all types — simpler deployment
- Schema validation catches VLM extraction errors structurally

**Negative:**
- VLM extraction is slower than YOLO (~3-5s vs ~50ms)
- VLM may hallucinate fields that don't exist in the document
- No bounding box information for non-YOLO types (VLM returns text only)

**Risks:**
- VLM prompt quality directly affects extraction accuracy — prompts may need iterative refinement per doc type
