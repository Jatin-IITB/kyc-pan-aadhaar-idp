# ADR-021: Evaluation-First Roadmap (Truth Engine before features)

**Date:** 2026-08-11
**Status:** Accepted
**Deciders:** Jatin Gupta

## Context

Phases 0–10 delivered the full pipeline, dashboard, and HITL tooling. The project's
headline metrics (94% field F1, 97% tamper recall, 88% auto-clear, p95 < 2.0 s) are
currently *targets*, not measurements — there is no dataset, no harness, and no CI
gate that produces them. Meanwhile the copy-move detector was scoring 1.0 on genuine
PAN cards, which went unnoticed for weeks precisely because nothing measured false
positives systematically. Two candidate directions for the next phase: add more
visible features (live DAG view, fraud rings, review copilot) or build the evaluation
substrate first.

## Decision

Adopt an evaluation-first roadmap (docs/ROADMAP.md, Phases 11–17). Phase 11 ("Truth
Engine") ships before any new feature work:

1. **Identity Forge** — synthetic Indian ID generator (valid PAN structure, Verhoeff
   Aadhaar check digits) emitting images + ground truth + YOLO labels
2. **Tamper Forge** — 6-class parameterized forgery generator to red-team our own
   forensics suite
3. **Eval harness** — `make eval` producing field F1, per-attack recall, spoof
   leakage, latency percentiles; CI-gated via `config/eval_thresholds.yaml`

Feature phases (12–16) all consume Phase 11 artifacts: synthetic data trains Phase 15
models, the harness gates Phase 14 agent claims, the load rig extends it in Phase 16.

## Consequences

**Positive:** every resume claim becomes reproducible; regressions (like the copy-move
false positive) surface in CI instead of in demos; synthetic data solves the
no-real-PII-dataset problem inherent to KYC; free YOLO training data.

**Negative:** delays visible feature work by one phase; synthetic documents are not
distribution-identical to real captures, so metrics carry a "on synthetic benchmark"
qualifier — stated honestly, this is still far stronger than unmeasured claims.
