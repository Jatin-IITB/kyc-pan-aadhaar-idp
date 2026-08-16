# ADR-029 — W5: Dual-Path Extraction Restoration

**Status:** Accepted  
**Date:** 2026-08-16  
**Decision Makers:** Jatin Gupta

## Context

The KYC pipeline has a dual-path extraction architecture:
1. **Fast path:** YOLO field detection → PaddleOCR → normalized extraction
2. **VLM fallback:** When YOLO confidence < 0.60, send image to VLM (minicpm-v)

Since W1-W4, the pipeline ran VLM-only because no trained YOLO models existed
at `models/yolov8/*/best.pt` — the pipeline loader fell back to `_NullDetector`
(returns no detections, forces VLM path for every document).

W5 restores the fast path by sourcing real-world-trained YOLO models.

## Decision

### D1: Pre-trained Aadhaar model from HuggingFace

Use `arnabdhar/YOLOv8-nano-aadhar-card` — a YOLOv8n model pre-trained on
real Aadhaar card images with 5 field classes:

| Model class     | Pipeline field   |
|-----------------|------------------|
| AADHAR_NUMBER   | aadhaar_number   |
| NAME            | name             |
| DATE_OF_BIRTH   | dob              |
| GENDER          | gender           |
| ADDRESS         | address          |

Published metrics: mAP@50 = 0.963, mAP@50-95 = 0.748.

**Why not train our own?** The pre-trained model already exceeds our quality
bar on real data. Training our own would require sourcing and annotating
hundreds of Aadhaar images with no clear quality improvement.

### D2: PAN model trained on Roboflow real data

No pre-trained PAN field detector exists at comparable quality. Train YOLOv8n
on the `pancard-info-detection` dataset from Roboflow Universe (annotated
real PAN card images with field-level bounding boxes).

Alternative datasets evaluated:
- `pan-1.8k` (Roboflow, 712 images)
- `pan_card_detection` (Roboflow, April 2025)

The `pancard-info-detection` by DocumentVerification was chosen for annotation
quality and field coverage.

### D3: Field name mapping in FieldDetector

Pre-trained models use their own class names. Added `field_map: Dict[str, str]`
parameter to `FieldDetector` that maps model class names to our pipeline's
expected field names. Configured per-model in `config/models.yaml`.

Mapping is applied at detection time (`_map_label`), so downstream code
(PaddleOCR, normalization, validation) sees consistent field names regardless
of which model produced the detections.

### D4: No synthetic training data

User explicitly rejected training on Identity Forge synthetic output:
> "you can get lot of training/testing data from kaggle online for free"

Real annotated datasets from Kaggle, Roboflow Universe, and HuggingFace
provide better domain coverage than our synthetic renderer, which has
limited template variety and unrealistic visual properties.

### D5: PaddleOCR already integrated

`services/ocr_paddle/roi_ocr.py` (ROIOCR class) was already implemented and
PaddleOCR is installed in `.venv`. The OCR pipeline includes:
- CLAHE contrast enhancement for small crops
- Field-name-aware text cleaning (date patterns, PAN format, Aadhaar digits)
- Per-field confidence scoring and collapse-to-best

No changes needed — wiring YOLO models in activates the full fast path.

## Data Sourcing

| Doc type | Source | Type | Size | Auth needed |
|----------|--------|------|------|-------------|
| Aadhaar  | HuggingFace `arnabdhar/YOLOv8-nano-aadhar-card` | Pre-trained model | 6MB | None |
| PAN      | Roboflow `pancard-info-detection` | Annotated dataset | ~500 images | Free API key |
| PAN alt  | Roboflow `pan-1.8k` | Annotated dataset | 712 images | Free API key |

Download script: `tools/train/download_pretrained.py`
Training script: `tools/train/train_yolo_fields.py`

## Pipeline Flow (after W5)

```
Document image
    │
    ├─→ YOLO field detection (YOLOv8n, per doc type)
    │       │
    │       ├─→ [confidence ≥ 0.60] → PaddleOCR on detected ROIs → normalize
    │       │                                                          │
    │       └─→ [confidence < 0.60] → VLM extraction (minicpm-v) ─────┤
    │                                                                  │
    │                                                         ensemble node
    │                                                              │
    ├─→ forensics (parallel) ──────────────────────────────→ decide node
    │                                                              │
    └─→ validate → policy verify → cross-doc → LLM rescue → audit commit
```

Fast path latency: ~200ms (YOLO ~50ms + PaddleOCR ~150ms)
VLM fallback latency: ~2-4s

## Limitations

1. **Aadhaar model coverage:** Pre-trained model has 5 classes — does not
   detect `photo` or `vid` (Virtual ID). VLM fallback covers these fields.

2. **PAN model quality unknown:** Until trained and evaluated, PAN field
   detection accuracy is TBD. The VLM fallback provides a safety net.

3. **Field mapping brittleness:** If a model retraining changes class names,
   `config/models.yaml` field_map must be updated. Unmapped classes are now
   logged and dropped (post-audit fix), but no startup validation that mapped
   names match schema expectations.

4. **No cross-model ensemble yet:** The ensemble node picks YOLO or VLM
   result — it does not merge field-level best from both paths. This is
   a future improvement opportunity.

5. **VLM fallback threshold on wrong scale (pre-existing):**
   `_yolo_confidence_router` in `workflow.py` compares `yolo_confidence`
   against 0.60, but `yolo_confidence` is a composite score on a 0-4 scale
   (`2*is_valid + coverage + avg_conf`), not a 0-1 confidence. The 0.60
   threshold almost never triggers VLM fallback. This is a pre-existing
   design issue (predates W5) that should be addressed when calibrating
   the ensemble path.

## Environment Constraints

### E1: Roboflow SDK rejected — OpenCV conflict
`pip install roboflow` pins `opencv-python-headless==4.10.0.84`. The venv already
carries `opencv-python 5.0.0.93` and `opencv-contrib-python 4.10.0.84`; a third
`cv2` would likely shadow the GUI-capable build the pipeline depends on, and it
downgrades `idna` 3.18 → 3.7.

Instead, `fetch_roboflow_dataset()` in `tools/train/download_pretrained.py` calls
Roboflow's REST API using stdlib `urllib`. Net new dependencies: zero.

### E2: Corporate TLS interception (Zscaler)
The dev machine sits behind a Zscaler TLS-inspecting proxy. Observed:

| Host | Issuer | Result |
|---|---|---|
| `huggingface.co` | Amazon RSA 2048 | Not intercepted — works |
| `api.roboflow.com` | Zscaler Intermediate Root CA | Intercepted |

Python 3.13+ sets `VERIFY_X509_STRICT` in `ssl.create_default_context()` by
default. Zscaler's root CA violates RFC 5280 (Basic Constraints not marked
critical), so strict verification rejects it:

```
SSLCertVerificationError: certificate verify failed:
Basic Constraints of CA cert not marked critical
```

`_ssl_context()` clears **only** that flag. Full chain-of-trust and hostname
verification against the system trust store are retained — verification is not
disabled. `SSL_CERT_FILE` can override the CA bundle if a custom one is needed.

Note this is environment-specific: CI runners without TLS interception are
unaffected, and the cleared flag is a no-op there.

## Audit Findings (2026-08-16)

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| S1 | Significant | `photo`/`signature` in field_map break schema validation | Fixed: non-text fields now dropped by FieldDetector |
| S2 | Significant | VLM fallback threshold on wrong scale (pre-existing) | Documented in limitations |
| S3 | Significant | No _NullDetector test | Fixed: added test |
| S4 | Significant | Runtime `pip install roboflow` is unsafe | Fixed: replaced with clear error message |
| S5 | Significant | Unmapped labels silently pass through | Fixed: unmapped labels dropped + logged |
| S6 | Significant | Hardcoded HuggingFace filename | Fixed: dynamic .pt file discovery |

## Measured Results (2026-08-16)

| Model | Source | mAP@50 | mAP@50-95 |
|---|---|---|---|
| Aadhaar | Pre-trained (HuggingFace) | 0.963 (upstream-reported) | 0.748 (upstream-reported) |
| PAN | Trained here, YOLOv8n, 50 epochs | 0.919 | 0.643 |

PAN per-class on the validation split:

| Class | P | R | mAP@50 |
|---|---|---|---|
| `name` | 0.992 | 1.000 | 0.995 |
| `fathername` | 0.933 | 1.000 | 0.995 |
| `dob` | 1.000 | 0.806 | 0.931 |
| `pan` | 0.901 | 0.600 | 0.757 |

**These PAN numbers are provisional and should not be quoted as-is.** The
Roboflow `pancard-info-detection` v1 dataset yielded only 71 training and 6
validation images (23 instances). An aggregate mAP computed over 6 images has
very wide error bars. Worse, the aggregate masks the operationally important
result: `pan` — the PAN number itself, the primary key of the document — has
the *lowest* recall at 0.60. Training converged (box loss 2.21 → 0.93) so the
limit is data volume, not schedule.

Next step for a credible PAN detector: combine multiple Roboflow PAN datasets
(`pan-1.8k`, `pancard-info-detection`, `pan_card_detection`) into a single
training set with reconciled class names, targeting ≥500 annotated images.

### E3: field_map / model-class drift (caught during this work)

The trained PAN model emits classes `dob`, `fathername`, `name`, `pan`. The
field_map written *before* training anticipated `PAN_NUMBER`, `FATHER_NAME`,
`DOB`, `NAME` — none of which matched. Because D3's unmapped-label handling
drops rather than passes through, **the PAN detector produced zero fields** and
the pipeline silently degraded to VLM-only.

This failure mode is invisible at runtime: no exception, no error metric — just
an empty detection list and a quiet fallback. `TestFieldMapCoversInstalledModels`
in `tests/unit/test_field_detector.py` now asserts every class a locally-present
model emits has a field_map entry, and skips when weights are absent so CI stays
green. Verified to fail against the broken mapping before being committed.

Both detectors were then confirmed end-to-end to emit correctly-mapped pipeline
field names (`pan_number`, `father_name`, `dob`, `name`, `aadhaar_number`,
`address`).
