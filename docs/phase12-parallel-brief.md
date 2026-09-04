# Phase 12 — Evidence Engine (parallel workstream brief)

**Status:** Proposed
**Runs parallel to:** Phase 11 W16+ (forensics, owned by the other agent)
**Reserved ADR range:** 050–069 (Phase 11 keeps 040–049)

---

## Thesis

Phase 11 made the *forensics* claims falsifiable. Every other headline claim in
the README is still unmeasured. Phase 12 does to the rest of the system what
Phase 11 did to forensics: build the harness that could prove the claim wrong,
run it, and publish whatever it says.

Concretely, four claims currently have no evidence behind them:

| README claim | Evidence today |
|---|---|
| "Citation-grounded regulatory compliance" via hybrid RAG | **None.** 709 lines across 4 modules, 0 tests, 0 metrics. |
| "Confidence calibration" | **None.** Hand-set weights + hand-set temperature. No ECE. |
| Dual-path extraction (YOLO+OCR fast path) | **None.** F1 measured for the VLM tier only. |
| 13-node LangGraph pipeline | **No latency.** Per-detector p95 only; no full-graph number. |

The project's own history says this matters: ADR-028 found the `text_splice`
100% was tautological *because* someone built the harness that could expose it.
Unmeasured subsystems are where the next tautology is hiding.

---

## Hard rule: no circular evaluation

This is the failure mode this project has already been bitten by once.

- **Do not** build the RAG golden set by running the retriever and labelling
  what it returns. That measures self-consistency, not retrieval.
- **Do not** fit the calibrator on the same split you report ECE on.
- **Do** mirror ADR-030's discipline: disjoint fit/report splits, leave-one-out
  where the sample is small, and a content hash binding any fitted artifact to
  the data it was fit on.
- **Do** write the negative result if the ablation says a component doesn't earn
  its complexity. ADR-019 (rotation classifier scored 0/4, was disabled not
  shipped) and ADR-033 (DCT copy-move negative result) are the precedent. A
  workstream that concludes "the cross-encoder adds nothing, remove it" is a
  success, not a failure.

---

## File ownership (prevents merge collisions)

Phase 11 (other agent) owns:

    services/forensics/**
    tools/forge/**
    config/font_profiles.json
    tests/unit/test_forensics.py
    docs/adr/04*.md
    config/eval_thresholds.yaml  →  forensics: and decision: blocks ONLY

Phase 12 (this workstream) owns:

    services/rag/**
    services/decisioning/**
    services/cross_doc/**
    services/graph/**            (timing instrumentation only)
    services/extraction/**
    tools/eval/**                (new modules; coordinate on run_eval.py)
    config/policies/**
    tests/unit/test_rag_*.py, test_calibration_*.py, test_extraction_*.py
    docs/adr/05*.md, 06*.md
    config/eval_thresholds.yaml  →  extraction:, NEW rag:, NEW calibration: blocks ONLY

Shared, needs care:

- **`config/eval_thresholds.yaml`** — both sides add gates. Stay strictly inside
  your own top-level blocks; YAML conflicts then stay trivially resolvable.
- **`tools/eval/run_eval.py`** — Phase 12 adds new stages. Append new stage
  functions at the end of the file rather than editing existing ones, and add
  new CLI flags rather than changing existing flag semantics.
- **`README.md`** — **do not touch.** Record what should change in your ADR
  under a "README deltas" heading. A single merge pass updates it at the end.
  This is the highest-collision file in the repo.

---

## WS-A — RAG evaluation harness

**The largest gap in the project.** `services/rag/` is 709 lines with zero tests
and zero quality measurement. The README advertises dense + BM25 + RRF +
cross-encoder reranking — four stages of complexity, none of it justified by a
number.

### A1. Golden question set

Build 60–80 questions over `config/policies/` with labelled relevant sections.

Derive labels from the **policy text structure**, not from retriever output.
Each policy doc is already `## Section`-delimited and the indexer chunks on those
boundaries — so a question authored against a specific section has a
ground-truth chunk id by construction.

Cover deliberately: single-section lookups, multi-section synthesis (e.g. a
question needing both the PMLA PAN rule and the RBI OVD definition), negative
questions with no supporting section (the system should abstain, not
hallucinate a citation), and near-miss distractor questions where lexical
overlap points at the wrong section.

Store as `data/rag_eval/golden.jsonl`, split fit/report by question id hash.

### A2. Retrieval metrics

Implement `tools/eval/rag_metrics.py`: recall@1/5/10, MRR, nDCG@10.

### A3. Component ablation — the headline result

Run the same golden set through five configurations:

    dense only
    BM25 only
    dense + BM25, RRF fused
    RRF + cross-encoder rerank    (current shipped config)
    BM25 + cross-encoder rerank   (does dense earn its Qdrant dependency?)

This either justifies the architecture or identifies a stage to delete. Both
outcomes are publishable and both improve the project. A 91-line corpus is
small enough that BM25 may well match dense retrieval — if so, that is a real
finding about when hybrid retrieval is worth it, and it is a far more
interesting thing to be able to discuss than an unmeasured pipeline.

### A4. Citation faithfulness

Retrieval recall is not the claim — *citation-grounded* is. For each verdict
from `policy_verifier.py`, check whether the cited chunk actually entails the
verdict. Use an LLM judge (qwen3:8b) with a rubric, then hand-verify a 20-case
sample to measure judge agreement. Report both numbers; a judge you haven't
validated is another unmeasured layer.

Target metric: `citation_support_rate`. Expect this to be the least flattering
number in the project. Publish it anyway.

### A5. Gates

Add a `rag:` block to `eval_thresholds.yaml` with the measured floor, matching
the ratcheting convention.

---

## WS-B — Real confidence calibration

`services/decisioning/calibrator.py` applies fixed weights
(0.35/0.25/0.25/0.15) and a fixed temperature (1.5). Neither was fit. Nothing
measures whether the resulting 0–1 number is a calibrated probability. Calling
this "calibration" is currently a naming claim, not a measured property.

### B1. Outcome labels

The synthetic corpus already knows ground truth: genuine vs forged, and
extraction correct vs incorrect against `truth/*.json`. That yields
(feature_vector → binary outcome) pairs without any new labelling. Build the
dataset from the **tuning** split; report on **holdout**.

### B2. Measure the status quo first

Before changing anything, compute ECE, MCE, Brier score, and a reliability
diagram for the current hand-set config. This is the baseline the change has to
beat, and it is the number that tells you whether T=1.5 was lucky or wrong.

### B3. Fit properly

Fit temperature on the tuning split (standard temperature scaling — single
parameter, minimises NLL). Then ablate the four weights: does the 0.35/0.25/
0.25/0.15 split beat uniform weighting? Does logistic regression on the four
scores beat both? Keep the simplest config that wins, and say so.

### B4. Gates + ADR

Add `calibration: ece_max` to the thresholds. Write the ADR with the reliability
diagram. If fitted temperature turns out close to 1.5, that is a fine result —
report that the original guess was defensible and now it is *measured*.

---

## WS-C — Fast path + full-graph latency

Both are explicitly listed as known gaps in the README. Both are now unblocked:
PAN and Aadhaar YOLO weights are present at
`models/yolov8/{pan,aadhar}_field_detector_v1/best.pt`, and Ollama is up.

### C1. Per-path extraction F1

Today `run_eval.py` measures the ensemble output only. Add a path selector so
the harness can score:

    YOLO+OCR only
    VLM only          (current measured number: 94.2% micro / 96.8% fuzzy)
    ensemble          (currently conflated with VLM-only in reporting)

This directly tests the `pick_best()` weighting in
`services/extraction/ensemble.py:26-27` (`yolo*0.6` vs `vlm*0.4`). Those
constants are unjustified today. If the ensemble does not beat both components,
that is a finding.

Note the known weakness this will expose: PAN detector `pan`-class recall is
0.60 on a 6-image validation set. The fast path will likely score poorly on the
single most important field. Report it plainly — it is already disclosed in the
README and confirming it with a real number strengthens rather than weakens the
project.

### C2. Extraction on holdout

Extraction currently runs on the tuning split only, sampling 12 docs. Extend to
the holdout split and raise the sample to the full 30/split. Without this,
extraction F1 has no held-out number at all — the exact criticism the forensics
work exists to avoid.

### C3. Full-graph latency

`tools/profile_latency.py` measures a mean over 10 runs and is not wired into
CI. Add per-node timing to the LangGraph nodes, emit a per-node p50/p95
breakdown plus an end-to-end number, and wire it into the eval report. This
closes the "full-graph p95 latency not benchmarked" gap and gives a per-node
profile that shows *where* the time goes — more useful than a single figure.

---

## WS-D — Stretch, only after A–C land

- **Policy corpus depth.** 91 lines across 2 documents is a toy corpus; real RBI
  Master Direction is ~200 pages. Expanding it makes WS-A's retrieval numbers
  mean something at realistic scale, and will likely change the dense-vs-BM25
  conclusion. Do this *after* A3 so you have a before/after.
- **Cross-doc threshold validation.** The 0.85/0.65 name-match thresholds in
  `entity_resolver.py:101-138` are unvalidated. Generate a labelled pair set
  (transliteration variants, OCR confusions like O/0 and I/1, married-name
  changes, genuinely-different people) and measure precision/recall at
  threshold. Apply the interior-point selection discipline from
  `feedback_threshold_selection` — pick an interior point, not the first value
  that clears the constraint.
- **Integration test.** No end-to-end test covers RAG → cross-doc → decisioning.

---

## Sequencing

WS-A and WS-B are independent and can run concurrently. WS-C touches
`run_eval.py`, which is the coordination point with Phase 11 — do it third, or
first if the parallel agent wants a quick win before the RAG golden set is
ready. WS-D is gated on A–C.

Suggested order: **A1→A2→A3 → B1→B2→B3 → C1→C2→C3 → A4→A5→B4 → D**.

Ship each workstream as its own commit with an ADR, following the Phase 11
convention: implement → audit with a separate agent pass → remediate → commit.

---

## Definition of done

Every claim in the README's "Why this is interesting" table has a number behind
it, a harness that reproduces the number, and a CI gate that goes red if it
regresses. Including the numbers that are unflattering.
