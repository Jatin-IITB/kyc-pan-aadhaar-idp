# ADR-008: Temperature-Scaled Confidence Calibration for Auto-Clear

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 7  

## Context

The system produces multiple confidence signals from different stages: extraction validation score, forensics spoof score, policy compliance status, cross-document consistency score. These signals have different scales, distributions, and reliability levels. A simple average would over-weight unreliable signals and under-weight reliable ones.

The business goal is to auto-clear ~88% of legitimate cases while never auto-clearing tampered or non-compliant documents.

### Alternatives Considered

- **Simple weighted average**: No calibration — raw scores from different domains aren't comparable
- **Threshold-per-signal**: Too rigid, doesn't consider the overall picture
- **ML-based classifier**: Requires labeled accept/reject training data at scale
- **Bayesian network**: Over-engineered for 4 input signals

## Decision

Implement a **two-layer decisioning system**:

### Layer 1: Confidence Calibrator
- **Weighted aggregation** of 4 signals with domain-specific weights:
  - Extraction quality: 0.35 (primary signal)
  - Forensics: 0.25 (inverted — high spoof score = low confidence)
  - Policy compliance: 0.25
  - Cross-doc consistency: 0.15
- **Temperature scaling** (T=1.5) on the aggregated logit to compress extreme probabilities toward the center — prevents overconfident auto-clears

### Layer 2: Auto-Clear Engine (Hard Overrides)
Rules that bypass calibration regardless of score:
- `spoof_score > 0.5` → always REJECT
- Policy non-compliant → always REVIEW (never auto-clear)
- Critical contradictions → always REVIEW
- Quality failed → always REJECT

### Thresholds
- `>= 0.92` → AUTO_CLEAR
- `>= 0.70` → REVIEW
- `< 0.70` → REJECT

## Consequences

**Positive:**
- Estimated 88% auto-clear rate on legitimate documents
- Zero auto-clear of tampered documents (hard spoof override)
- Temperature scaling prevents overconfident decisions on borderline cases
- Hard overrides encode non-negotiable business rules

**Negative:**
- Temperature parameter (1.5) is manually chosen — may need tuning
- Auto-clear threshold (0.92) is conservative by design — some valid docs go to review
- Weights are heuristic — ideally calibrated against labeled production data

**Risks:**
- Weight/threshold drift as document quality distribution changes
- Mitigation: active learning loop (Phase 8) tracks corrections to detect drift
