# ADR-025: Eval Harness with Ratcheting CI Gates

**Date:** 2026-08-12
**Status:** Accepted
**Deciders:** Jatin Gupta

## Context

W1 gave us labeled genuine documents, W2 gave us labeled forgeries and the
first honest recall numbers. What remained was institutionalizing the
measurement so regressions surface in CI, not in demos (the copy-move false
positive survived for weeks precisely because nothing measured it).

## Decision

`make eval` runs `tools/eval/run_eval.py` — four tiers, one command,
reproducible from a clean checkout (datasets regenerate from fixed seeds):

1. **Forensic sweep** — all 5 detectors + SpoofScorer over every genuine and
   forged document: genuine FPR, per-attack recall matrix, per-detector
   latency p50/p95.
2. **Decision layer** — every measured spoof score pushed through the REAL
   `ConfidenceCalibrator` + `AutoClearEngine` (extraction/policy/cross-doc
   held at 1.0 to isolate the forensic gate).
3. **VLM extraction** (optional, needs Ollama; `--no-extraction` for CI) —
   sampled synthetic docs extracted and scored against ground truth: per-field
   exact + fuzzy F1 (Jaro-Winkler ≥ 0.90, reusing the Phase 6 implementation).
   Number/date fields never match fuzzily — one wrong digit is a different
   identity. Warm-up call first; per-sample failures skip, not abort.
4. **Gates** — `config/eval_thresholds.yaml` compared against measured
   metrics; `--check` exits non-zero on any FAIL. Tiers that did not run are
   SKIPPED, not failed. Outputs `eval/metrics.json` + self-contained
   `eval/report.html`.

**Gates encode the measured floor, not aspirations.** The build is green today
and goes red the moment a change degrades a certified metric; floors ratchet
up only when the report proves an improvement.

## First-run finding: 41 blind-spot auto-clears

The harness's first full run surfaced a real end-to-end exposure: 41 of 180
forgeries would AUTO_CLEAR. Dissection split the metric in two:

- **`flagged_leakage` = 0** — anything forensics flags (spoof ≥ 0.15 after
  calibration) never auto-clears. The decision layer is correct.
- **`undetected_autoclear` = 41** — forgeries forensics cannot see (font_swap
  20, regenerate 11, text_splice 9, screen 1) arrive with spoof ≈ 0,
  indistinguishable from genuine, and clear. This is the end-to-end price of
  the W4 recall gap, now measured. Its gate is a ceiling (45) that ratchets
  down as W4 closes each blind spot.

Conflating these two numbers would have either hidden a decision-layer bug
class (gate too loose) or made the gate permanently red (gate impossible).

## Consequences

**Positive:** every headline metric is now one command away and CI-enforced;
the W4 backlog has an end-to-end cost attached (41 clears), not just recall
percentages; `eval-fast` runs without any model dependency.

**Negative:** the extraction tier depends on local Ollama and ~30-60 s/doc VLM
latency, so CI certifies forensics/decision only; extraction F1 is certified
on developer machines until a hosted runner with a model exists. Metrics carry
the ADR-023 "synthetic benchmark" qualifier.
