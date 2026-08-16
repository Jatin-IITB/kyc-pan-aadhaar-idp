# KYC-IDP — Multi-Agent Document Intelligence for Indian KYC

A production-shaped document intelligence platform for Indian identity documents. Beyond extraction, it does **tamper forensics**, **citation-grounded regulatory compliance**, **cross-document entity resolution**, and **calibrated auto-clear decisioning** — with every stage emitting hash-chained audit events.

Built as a LangGraph state machine with dual-path extraction (YOLOv8 + PaddleOCR fast path, vision-LLM fallback) and a measurement harness that gates CI on ratcheting quality thresholds.

![Python](https://img.shields.io/badge/python-3.14-blue) ![Tests](https://img.shields.io/badge/tests-104-green) ![ADRs](https://img.shields.io/badge/ADRs-30-blueviolet) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Why this is interesting

Most document-extraction projects stop at "OCR the fields." The hard parts of real KYC are everything downstream:

| Problem | Approach here |
|---|---|
| Is this document *forged*? | 5-signal forensics suite fused with noisy-OR aggregation |
| Does this packet satisfy *RBI regulation*? | Hybrid RAG (dense + BM25 + RRF + cross-encoder) with per-requirement citations |
| Do the documents *agree with each other*? | Jaro-Winkler / Soundex entity resolution + Indian address normalization |
| Can we *safely* auto-approve? | Confidence calibration with hard-override rules |
| Can we *prove* what happened? | SHA-256 hash-chained audit ledger with replay |
| Are the metrics *real*? | Synthetic forge + held-out eval + CI gates that ratchet |

---

## Architecture

```
                        ┌────────────────────────────────────┐
 Client ──▶ FastAPI ──▶ │      LangGraph State Machine       │
                        │                                    │
                        │  ingest → quality_gate             │
                        │       ├── reject ─────────────┐    │
                        │       └── classify            │    │
                        │            ├── extract_yolo   │    │
                        │            │    ├─ conf ok    │    │
                        │            │    └─ VLM fallback│   │
                        │            │       ensemble   │    │
                        │            │       validate   │    │
                        │            │    policy_verify │    │
                        │            │      cross_doc   │    │
                        │            │     llm_rescue   │    │
                        │            └── forensics ─────┤    │
                        │                        decide │    │
                        │                  audit_commit │    │
                        └────────────────────────────────────┘
                                        │
           Postgres ── MinIO ── Qdrant ── Redis ── Streamlit Review UI
```

**13 graph nodes.** Forensics runs as a parallel branch — it never blocks extraction.

### Dual-path extraction

```
Document ──▶ YOLOv8 field detection
                ├── confidence OK  ──▶ PaddleOCR on ROIs ──▶ normalize   (~200ms)
                └── confidence low ──▶ Vision-LLM (minicpm-v)            (~2-4s)
                                            └──▶ ensemble picks best
```

The VLM path means document types without a trained detector still extract — the system degrades gracefully instead of failing.

---

## Measured results

All forensics numbers below come from `make eval`, scored on a **held-out seed pair** never used for tuning (seeds `777/888` vs tuning `42/123`). Reproduce with `make eval`.

### Forensics — held-out synthetic set

| Metric | Result |
|---|---|
| **Genuine false-positive rate** | **0/30 (0.0%)** |
| Overall tamper recall | 63.9% |
| Decision-layer leakage (flagged docs that auto-cleared) | **0** |
| Genuine auto-clear rate | 100% |

Per attack class:

| Attack | Recall | Detection basis |
|---|---|---|
| `exif_edit` | 100% | EXIF software tags, date anomalies |
| `text_splice` | 100% \* | JPEG quantization-table quality |
| `regenerate` | 100% \* | JPEG quantization-table quality |
| `screen_recapture` | 40% | FFT Moiré analysis |
| `copy_move` | 43% | ORB keypoint matching (alignment-free) |
| `font_swap` | 0% | Not yet detectable — needs OCR-context forensics |

> **\* Read these two honestly.** The 100% is *tautological*: the detector fires on "saved at Q<88," which is an artifact of how the forge writes these attacks — not evidence of tampering. An attacker who re-saves at Q=92 evades it completely. Documented in full in [ADR-028](docs/adr/028-w4-forensic-precision.md). They're gated at 0.90 to catch regressions, **not** as a claim of adversarial robustness.

### Forensics latency (p95, per detector)

| Detector | p50 | p95 |
|---|---|---|
| metadata | 0.1 ms | 0.2 ms |
| font | 3.6 ms | 5.2 ms |
| ELA | 6.5 ms | 8.6 ms |
| screen recapture | 16.0 ms | 18.6 ms |
| copy-move (ORB) | 88.2 ms | 111.3 ms |

### Field detection

| Model | Source | mAP@50 | mAP@50-95 |
|---|---|---|---|
| Aadhaar (`YOLOv8-nano-aadhar-card`) | Pre-trained, HuggingFace | 0.963 † | 0.748 † |
| PAN (YOLOv8n, 50 epochs) | Trained here on Roboflow data | 0.919 ‡ | 0.643 ‡ |

† Published by the upstream model author; not independently re-measured here.

‡ **Treat as provisional.** The PAN detector was trained on 71 images and validated on **6** — far too small for the aggregate to be meaningful. Per-class results show the weakness the average hides:

| Class | Precision | Recall | mAP@50 |
|---|---|---|---|
| `name` | 0.992 | 1.000 | 0.995 |
| `fathername` | 0.933 | 1.000 | 0.995 |
| `dob` | 1.000 | 0.806 | 0.931 |
| **`pan`** (the PAN number) | 0.901 | **0.600** | **0.757** |

The single most important field is the worst-performing one — it misses 40% of PAN numbers on an already-tiny validation set. A real deployment needs a substantially larger annotated set before this detector carries traffic; the VLM fallback covers the gap today.

### Scope of these numbers

Stated plainly so they aren't over-read:

- Measured on **synthetic** documents from the built-in Identity Forge, not production traffic.
- The 0% genuine FPR holds against forge output (Q=92). Real submissions arrive via phone cameras, WhatsApp, and scanners at varied quality — real-world FPR is **unmeasured**.
- Decision-layer figures isolate the forensic gate (extraction/policy/cross-doc held at 1.0).

The eval harness exists precisely to keep these claims falsifiable — it's what surfaced the tautology above.

---

## Components

| Module | What it does |
|---|---|
| `services/graph/` | LangGraph state machine — 13 nodes, conditional routing, parallel forensics |
| `services/forensics/` | ELA, ORB copy-move, font analysis, JPEG/EXIF metadata, FFT screen-recapture, noisy-OR scorer |
| `services/rag/` | Policy indexer, hybrid retriever (dense + BM25 + RRF), cross-encoder reranker, citation-grounded verifier |
| `services/cross_doc/` | Entity resolution, contradiction detection, Indian address normalization |
| `services/decisioning/` | Confidence calibrator, auto-clear engine with hard overrides |
| `services/audit/` | SHA-256 hash-chained ledger, state replay |
| `services/active_learning/` | Ground-truth DB, retrain triggers, model registry, regression checker |
| `services/extraction/` | VLM extractor, ensemble scoring, normalizers, LLM cleaner |
| `tools/forge/` | Identity Forge (Verhoeff-valid synthetic IDs) + Tamper Forge (6 attack classes) |
| `tools/eval/` | Eval harness with ratcheting CI gates |

---

## Engineering practices

- **30 ADRs** in [`docs/adr/`](docs/adr/) — every non-obvious decision recorded, including the ones that *failed*: [ADR-019](docs/adr/019-rotation-classifier.md) documents a rotation classifier that scored 0/4 on real cards and was disabled rather than shipped.
- **Independent audit per phase** — each phase reviewed by a separate pass, findings tracked Critical/Significant/Minor and remediated before moving on.
- **Ratcheting CI gates** — [`config/eval_thresholds.yaml`](config/eval_thresholds.yaml) encodes the current measured floor. Any change that degrades a certified metric turns the build red.
- **Held-out evaluation** — tuning and holdout seed pairs are separate and CI-enforced, so numbers can't be tuned into existence.
- **104 unit tests.**

---

## Quick start

```bash
docker compose up          # Postgres, MinIO, Qdrant, Redis, API, worker
alembic upgrade head       # schema
```

API docs at `http://localhost:8000/docs`.

Local development:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/unit -q     # tests
make eval                                    # reproduce the forensics metrics above
```

Fetch the Aadhaar field detector:

```bash
.venv/bin/python -m tools.train.download_pretrained --type aadhaar
```

Train a PAN detector (needs a free Roboflow API key):

```bash
read -s ROBOFLOW_API_KEY && export ROBOFLOW_API_KEY
.venv/bin/python -m tools.train.download_pretrained --type pan
```

Without YOLO weights present the pipeline runs VLM-only — it still works, just slower.

---

## API

```bash
# Submit a document
curl -X POST http://localhost:8000/jobs -F "file=@pan_card.jpg" -F "doc_type=auto"

# Poll result
curl http://localhost:8000/jobs/{job_id}

# Batch submit
curl -X POST http://localhost:8000/batches -F "files=@doc1.jpg" -F "files=@doc2.jpg"
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| API | FastAPI + Uvicorn |
| Queue | Celery + Redis |
| Field detection | YOLOv8n (Ultralytics) |
| OCR | PaddleOCR |
| Vision LLM | minicpm-v via Ollama |
| Text LLM | Qwen 3 8B via Ollama |
| Vector store | Qdrant + BAAI/bge-small-en-v1.5 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Persistence | Postgres (SQLAlchemy async) + MinIO |
| Review UI | Streamlit |

---

## Project structure

```
apps/
  api/              FastAPI gateway, auth, SSE progress, Prometheus metrics
  workers/          Celery tasks, pipeline/graph loader
  review_ui/        Streamlit HITL console (bbox overlays, forensics evidence)
  common/           Settings, SQLAlchemy models, database
services/
  graph/            LangGraph state machine + 13 nodes
  forensics/        Tamper detection suite
  rag/              Policy retrieval + verification
  cross_doc/        Entity resolution, contradictions
  decisioning/      Calibration + auto-clear
  audit/            Hash-chained ledger + replay
  active_learning/  Ground truth, retrain triggers, model registry
  extraction/       VLM, ensemble, normalization, LLM cleaner
  card_crop_yolov8/ YOLOv8 field detection
  ocr_paddle/       PaddleOCR ROI extraction
config/
  schemas/          JSON Schema per document type
  policies/         RBI/PMLA regulatory corpus
  eval_thresholds.yaml   CI quality gates
tools/
  forge/            Identity Forge + Tamper Forge
  eval/             Eval harness
  train/            Model download + training
docs/adr/           30 architecture decision records
```

---

## Current state & roadmap

**Working:** LangGraph pipeline, forensics suite, RAG policy engine, cross-doc intelligence, calibrated decisioning, audit ledger, active-learning scaffolding, synthetic forge, eval harness with CI gates, both field detectors wired and emitting correctly-mapped fields.

**Known gaps — stated deliberately:**
- **PAN detector is under-trained** — 71 train / 6 val images; `pan` class recall 0.60. Needs a materially larger annotated set before it should carry traffic.
- `font_swap` undetected (0% recall) — needs OCR-context font forensics.
- `text_splice` / `regenerate` detection is metadata-only and evadable; robust detection needs frequency-domain work (double-JPEG ghosts, DCT histogram analysis, or a learned localizer).
- No production-traffic calibration — all forensics numbers are synthetic-set numbers.
- End-to-end extraction F1 and p95 latency not yet benchmarked across the full graph.

---

## License

MIT
