# Roadmap — From Pipeline to Platform

> **Vision:** Every number on the resume reproducible by one command. Every decision
> explainable to a regulator. Every forgery caught by a system that red-teams itself.

## Where we are (Phases 0–10, shipped)

- 13-node LangGraph DAG: ingest → quality → classify → dual-path extract (YOLO/VLM) →
  ensemble → validate → LLM rescue → forensics ∥ policy RAG → cross-doc → calibrate →
  decide → hash-chained audit
- FastAPI + Celery/Redis workers (solo pool, VLM-safe), Postgres/MinIO/Qdrant compose
- 5-signal forensics (ELA, copy-move, font, EXIF, Moiré), hybrid RAG over RBI/PMLA/SEBI
- Active-learning skeleton, model registry with hot reload, SSE progress
- Glassmorphic Next.js dashboard + Streamlit HITL review UI

## North-star metrics (what the Truth Engine must certify)

| Claim | Target | Certified by |
|---|---|---|
| Field extraction F1 | ≥ 94% | Phase 11 eval harness |
| Tamper recall | ≥ 97% @ ≤3% FPR | Phase 11 red-team set |
| Auto-clear rate | ≥ 88%, **0% spoof leakage** | Phase 11 decision eval |
| Latency | p95 < 2.0 s/case | Phase 16 load rig |
| Document types | 12+ Indian IDs | Phase 11 synthetic forge |
| False rejects | −45% vs rules-only baseline | Phase 11 A/B eval |

These are the resume numbers. Right now they are aspirations; the roadmap's first job
is to turn them into measurements.

---

## Phase 11 — The Truth Engine *(evaluation, synthetic data, red team)*

**The keystone phase. Everything downstream inherits credibility from this.**

- **Identity Forge** — synthetic document renderer: Faker(`en_IN`) identities, valid
  PAN format + Verhoeff-checksummed Aadhaar numbers, per-type templates (PAN, Aadhaar,
  DL, Voter ID, Passport-lite), augmentations (blur, rotation, perspective, JPEG,
  lighting). Every image ships with ground-truth JSON **and YOLO bbox labels** — free
  training data for Phase 15.
- **Tamper Forge** — programmatic forgery generator, 6 attack classes: text splice,
  copy-move, font substitution, screen recapture simulation (Moiré + perspective +
  glare), EXIF manipulation, full regeneration. Labeled by attack type and region.
  *We attack our own forensics suite before a fraudster does.*
- **Eval Harness** — `make eval` → field-level P/R/F1 (exact + fuzzy), per-attack
  recall matrix, forensic ROC curves, decision confusion matrix, latency percentiles,
  HTML report. CI gate: metrics below threshold fail the build.
- **Forensic precision pass** — copy-move v2 (shift-vector clustering — shipped),
  per-detector threshold sweeps driven by the harness instead of vibes.
- **Dual path restored** — PaddleOCR installed, YOLOv8n retrained on synthetic docs,
  weights promoted through the registry. The resume's "dual-path" claim becomes real
  on this machine again.

**Resume payoff:** "Built a self-red-teaming eval harness: synthetic Indian ID forge +
6-class tamper generator certifying 94% F1 / 97% tamper recall in CI."

**Exit criteria:** `make eval` prints the north-star table from a clean checkout.

## Phase 12 — The Living Pipeline *(real-time DAG theater)*

Wire per-node SSE events into the dashboard: the actual LangGraph rendered as an
animated graph, nodes lighting up as they execute, timings and evidence materializing
live. Confidence flows visibly from extraction → calibration → decision.

**Resume payoff:** the demo. Nobody forgets watching a document flow through a living
graph. **Exit:** upload a doc, watch all 13 nodes fire in real time.

## Phase 13 — The Identity Graph *(fraud rings)*

Every document embedded (image + face crop) into Qdrant at ingest. Cross-case
analytics: same face under different names → CRITICAL; same address across many
identities; submission velocity bursts. Linked-case graph view in the dashboard.

**Resume payoff:** "Detected fraud rings via cross-case embedding similarity" — a real
fintech capability, not a class project feature. **Exit:** submit the same face under
two names, watch the ring alert fire.

## Phase 14 — The Review Copilot *(agentic HITL)*

A case-aware agent in the review UI: answers "why was this flagged?", cites forensic
evidence and policy chunks inline, drafts adverse-action notices with RBI citations,
and can re-run individual forensic detectors on demand (tool use). Human reviewers get
a colleague, not a dashboard.

**Resume payoff:** agentic AI with tool use, grounded in a real workflow.
**Exit:** reviewer asks three natural-language questions, gets cited answers.

## Phase 15 — The Self-Improving Loop *(fine-tune & distill)*

- LoRA fine-tune a small VLM (Qwen2.5-VL-3B class) on Identity Forge output;
  benchmark against the zero-shot baseline in the eval harness
- Distill: big-VLM labels → fast student for the hot path
- Close the active-learning circle: reviewer corrections → retrain trigger → regression
  check → registry promote → hot reload. The skeleton exists; make it breathe.

**Resume payoff:** "Fine-tuned a VLM on 10k synthetic Indian IDs; +F1 at a fraction of
the latency." **Exit:** a fine-tuned model wins an A/B in the registry and serves live.

## Phase 16 — The Fortress *(compliance, privacy, scale)*

- **DPDP Act 2023** compliance mapping; Aadhaar masking (display last 4 only) per UIDAI
  regulations — deep domain credibility for Indian KYC
- PII vault: field-level encryption at rest, log-redaction middleware
- Locust load rig proving p95 < 2.0 s; horizontal worker scaling story
- OpenTelemetry traces per graph node → Grafana

**Resume payoff:** production maturity + regulatory depth interviewers rarely see.
**Exit:** load test report + compliance matrix in the repo.

## Phase 17 — The Showcase

Architecture diagrams, 90-second demo video, README as a landing page, technical blog
post ("How I taught a pipeline to doubt documents"). A flagship deserves a stage.

---

## Sequencing rationale

```
11 Truth Engine ──┬─► 12 Living Pipeline (demo uses certified pipeline)
                  ├─► 13 Identity Graph (eval measures ring detection)
                  ├─► 14 Review Copilot (agent cites certified evidence)
                  ├─► 15 Self-Improving (synthetic data trains the models)
                  └─► 16 Fortress (load rig extends the harness)
                                    └─► 17 Showcase (numbers are real)
```

Phase 11 is first because it is the only phase that makes every other phase *provable*.
Synthetic data trains Phase 15's models, the harness gates Phase 14's claims, and the
red team keeps Phase 13 honest. Features without evaluation are demos; features with
evaluation are engineering.
