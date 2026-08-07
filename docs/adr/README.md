# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the KYC-IDP Multi-Agent Document Intelligence Platform.

ADRs document significant architectural decisions made during the project, including context, rationale, and consequences.

## Format

Each ADR follows the standard format:
- **Status**: Proposed | Accepted | Deprecated | Superseded
- **Context**: The situation and forces at play
- **Decision**: What we decided and why
- **Consequences**: The resulting impact

## Index

| ADR | Title | Status | Phase |
|-----|-------|--------|-------|
| [001](001-langgraph-state-machine.md) | LangGraph state machine over monolithic pipeline | Accepted | 1 |
| [002](002-dual-path-extraction.md) | Dual-path YOLO+VLM extraction with ensemble | Accepted | 2 |
| [003](003-vlm-for-doc-expansion.md) | VLM-only extraction for non-YOLO document types | Accepted | 3 |
| [004](004-opencv-forensics.md) | Pure OpenCV/numpy forensics (no external APIs) | Accepted | 4 |
| [005](005-hybrid-rag-policy.md) | Hybrid RAG for policy compliance verification | Accepted | 5 |
| [006](006-fuzzy-entity-resolution.md) | Multi-algorithm fuzzy entity resolution | Accepted | 6 |
| [007](007-hash-chained-audit.md) | SHA-256 hash-chained immutable audit ledger | Accepted | 7 |
| [008](008-temperature-calibrated-decisioning.md) | Temperature-scaled confidence calibration for auto-clear | Accepted | 7 |
| [009](009-active-learning-loop.md) | File-based ground truth with retrain triggers | Accepted | 8 |
| [010](010-infra-postgres-minio-qdrant.md) | Postgres + MinIO + Qdrant infrastructure stack | Accepted | 0 |
| [011](011-contextvars-dependency-injection.md) | Python contextvars for graph node DI | Accepted | 1 |
| [012](012-prometheus-metrics.md) | In-process Prometheus metrics without external library | Accepted | 9 |
| [013](013-sse-progress-streaming.md) | Server-Sent Events for case progress | Accepted | 9 |
| [014](014-spoof-threshold-consolidation.md) | Consolidate spoof score thresholds into single module | Accepted | Audit |
| [015](015-ollama-chat-api.md) | Use Ollama /api/chat instead of hardcoded prompt tokens | Accepted | Audit |
