# ADR-019: Trained Rotation Classifier

**Status:** Accepted  
**Date:** 2026-08-08

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

- **Pros**: Single-pass rotation detection (~5ms vs ~200ms), works without YOLO weights, trained on 31K real document images
- **Cons**: Requires separate training step; model weights not in git (must retrain or download)
- **Training**: `python -m tools.train.train_rotation --epochs 10 --batch-size 64 --lr 1e-3`
