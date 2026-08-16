# ADR-019: Trained Rotation Classifier

**Status:** Superseded by the "Outcome" section below — model trained but **disabled in production**  
**Date:** 2026-08-08 (outcome recorded 2026-08-15)

## Context

The pipeline needed to detect document rotation (0°, 90°, 180°, 270°) before extraction. The previous approach brute-forced all 4 rotations through both PAN and Aadhaar YOLO detectors (8 inference passes), which was slow and didn't work when YOLO weights were unavailable (VLM-only mode defaulted to rot0).

A Kaggle rotation-angle-detection dataset (31,032 images, 4 balanced classes) was available for training a dedicated classifier.

## Decision

Train a MobileNetV3-Small classifier on the Kaggle dataset:
- **Architecture**: MobileNetV3-Small (2.5M params) — fast inference, small footprint
- **Input**: 224x224 RGB, ImageNet normalization
- **Output**: 4 classes → rot0, rot90, rot180, rot270
- **Training**: AdamW (lr=1e-3), CosineAnnealing, 10 epochs, 85/15 train/val split
- **Device**: Apple MPS (Metal Performance Shaders) for GPU acceleration

### Integration

The `RotationClassifier` in `services/doc_classifier/rotation_model.py`:
- Loads from `models/rotation_classifier/best.pt` at worker startup
- Falls back to rot0 if weights are missing
- Used by the `classify_node` before YOLO/VLM extraction
- Single inference pass (~5ms) replaces 8 YOLO passes (~200ms+)

### Dataset organization

Kaggle dataset moved from repo root to `data/datasets/rotation-angle-detection/` (under gitignored `data/`). Classes map: `not-rot→rot0`, `cw-90→rot90`, `cw-180→rot180`, `acw-90→rot270`.

## Consequences

- **Pros**: Single-pass rotation detection (~5ms vs ~200ms), works without YOLO weights
- **Cons**: Requires separate training step; model weights not in git (must retrain or download)
- **Training**: `python -m tools.train.train_rotation --epochs 10 --batch-size 64 --lr 1e-3 --pretrained`

## Outcome (2026-08-15) — model rejected, feature disabled

The v1 model was trained and **failed validation**. It is disabled via
`rotation.enabled: false` in `config/models.yaml`.

**What happened:**

1. Trained 10 epochs, reaching **74.2% held-out accuracy** on the Kaggle split.
   Well above the 25% chance baseline, but far short of what this task needs.
2. Spot-checked against a real PAN card at all four rotations: **0/4 correct**, and
   *confidently* wrong — rot180 was predicted as rot0 at 0.993 confidence.

**Root causes:**

- **Domain gap (primary).** The Kaggle rotation set is not PAN/Aadhaar cards. 74% in-
  distribution accuracy did not transfer at all to the documents this pipeline sees.
- **No pretrained weights.** The ImageNet download failed at training time and the
  `--pretrained` flag was never wired to argparse, so the run silently used random init.
- **Harmful augmentation.** `RandomHorizontalFlip` was in the training transform. Mirroring
  a rot90 document makes it look like rot270, injecting label noise into a task whose only
  signal is orientation. Removed.

**Why a confidence threshold is not sufficient on its own:** the model was wrong at 0.993
confidence, so thresholding alone cannot rescue a model with this domain gap. A confidence
gate (`rotation.min_confidence`, default 0.90) was added as defense-in-depth, but the
feature stays off until a model earns it.

**Bug this exposed:** `classify_node` originally overrode the detector-based rotation
search unconditionally and ignored the returned confidence entirely. Any loaded model won,
however bad. This shipped in `858c771` and was live on `main` for a week. Now the model must
be explicitly enabled *and* clear `min_confidence`, otherwise the detector search wins.

**Re-entry criteria.** Before flipping `rotation.enabled` back on, a model must:

1. Train with `--pretrained` on **in-domain** PAN/Aadhaar images, not the Kaggle set.
2. Pass `python -m tools.train.eval_rotation` with strong **per-class** accuracy — overall
   accuracy hides the rot90/rot270 confusion that matters most.
3. Score 4/4 on the real-PAN rotation spot-check that v1 failed.

**Dataset note.** The 6.2GB Kaggle dataset at `data/datasets/rotation-angle-detection/` was
deleted at some point during the week of 2026-08-08→15. It is under gitignored `data/` and
is not recoverable from git; re-download from Kaggle to retrain.
