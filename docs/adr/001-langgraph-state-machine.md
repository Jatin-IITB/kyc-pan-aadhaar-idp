# ADR-001: LangGraph State Machine Over Monolithic Pipeline

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 1  

## Context

The original `KYCPipeline.extract_from_bgr()` was a ~486-line monolithic method that combined quality gating, classification, rotation search, YOLO detection, OCR, normalization, validation, and LLM rescue in a single call chain. This made it impossible to:

- Add parallel execution paths (e.g., forensics alongside extraction)
- Insert new processing stages without modifying the core method
- Visualize or debug the processing flow
- Test individual stages in isolation

We needed an orchestration approach that was:
1. Composable — easy to add/remove/reorder nodes
2. Observable — visualizable flow, per-node state inspection
3. Compatible — same output format as the existing pipeline

### Alternatives Considered

- **Prefect / Airflow**: Too heavy for in-process orchestration, designed for batch DAGs not per-request processing
- **Custom DAG class**: Reinventing what LangGraph already provides; no tooling ecosystem
- **Simple function chain**: No conditional routing, no parallelism, hard to add forensics branch

## Decision

Decompose the monolithic pipeline into a **LangGraph `StateGraph`** with:

- **`CaseState` TypedDict** as the shared state flowing through all nodes
- **13 nodes**: ingest, quality_gate, classify, extract_yolo, extract_vlm, ensemble, validate, policy_verify, cross_doc, forensics, llm_rescue, decide, audit_commit
- **Conditional routing**: quality gate (reject vs continue), cross-doc (multi-doc vs single), rescue (invalid vs valid)
- **Parallel branches**: forensics runs alongside extraction (both start from classify)

Each node wraps existing logic from `pipeline.py`, maintaining output compatibility.

## Consequences

**Positive:**
- New stages (forensics, policy, cross-doc) added without touching existing nodes
- Forensics runs in parallel with extraction — no latency penalty
- Graph is visualizable via `graph.get_graph().draw_mermaid()`
- Individual nodes are unit-testable

**Negative:**
- Added dependency on `langgraph` library
- `CaseState` TypedDict must be updated when adding new fields (25+ fields now)
- Slight overhead from state copying between nodes (mitigated by dict merge, not deep copy)

**Risks:**
- LangGraph is relatively new (v0.0.x) — API may change
- Mitigation: nodes are plain functions with no LangGraph imports; only `workflow.py` depends on the library
