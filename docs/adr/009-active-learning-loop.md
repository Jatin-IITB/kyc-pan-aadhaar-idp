# ADR-009: File-Based Ground Truth with Retrain Triggers

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 8  

## Context

Human reviewers correct extraction errors during the HITL review process. These corrections represent high-quality labeled data that should improve the system over time. Without a feedback loop, the same errors repeat indefinitely.

### Alternatives Considered

- **Direct database storage only**: Harder to export for training, mixes operational and training data
- **Full MLOps platform (MLflow, Weights & Biases)**: Heavy dependency for a single-model pipeline
- **Manual retraining process**: Error-prone, no automation, no regression detection
- **Online learning (continuous model update)**: Risky without regression checks, hard to roll back

## Decision

Implement a **file-based active learning loop** with four components:

1. **GroundTruthDB**: JSONL append-only store for corrections. Each record captures document_id, doc_type, field_name, original_value, corrected_value, reviewer, extraction_path. Supports training set export and error distribution analysis.

2. **RetrainTrigger**: Fires when ANY of:
   - Correction count >= 100 (volume threshold)
   - Field F1 drops >= 0.02 from baseline (quality drift)
   - Concentrated errors: a single doc_type:field pair has >= 50 corrections

3. **ModelRegistry**: JSON manifest at `models/registry.json` tracking versions with status (registered → active → retired). Supports promote/rollback. Hot-reload in pipeline_loader clears LRU cache when active version changes.

4. **RegressionChecker**: Runs new model against held-out eval set, computes per-field F1, compares against baseline. Blocks promotion if regression detected (F1 drop > 0.02 on any field).

Review UI adapter (`save_review()`) ingests corrections to GroundTruthDB automatically.

## Consequences

**Positive:**
- Corrections automatically feed into training pipeline
- Regression checker prevents deploying worse models
- Hot-reload enables zero-downtime model updates
- Error distribution analysis identifies systematic extraction failures

**Negative:**
- JSONL file grows unbounded — needs periodic archival for large volumes
- Retrain trigger fires but doesn't actually retrain — requires external training pipeline
- Held-out eval set must be manually curated and maintained

**Risks:**
- Correction quality depends on reviewer accuracy — garbage in, garbage out
- Mitigation: reviewer field enables filtering by reviewer quality
