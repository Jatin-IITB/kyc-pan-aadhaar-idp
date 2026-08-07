# ADR-011: Python contextvars for Graph Node Dependency Injection

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 1  

## Context

LangGraph nodes are plain functions with signature `(state: CaseState) -> CaseState`. They need access to shared resources (YOLO detectors, OCR engine, classifier, LLM cleaner, VLM extractor) but LangGraph doesn't support passing extra arguments to node functions.

### Alternatives Considered

- **Global singletons**: Works but makes testing hard, no isolation between concurrent invocations
- **Closure-based nodes**: `def make_node(deps): def node(state): ...` — verbose, breaks typing
- **State injection**: Put deps into CaseState — pollutes the state with non-data objects (numpy arrays, models)
- **Class-based nodes**: `class IngestNode: def __call__(self, state):` — more boilerplate, LangGraph expects plain functions

## Decision

Use Python's **`contextvars`** module for dependency injection:

```python
@dataclass
class PipelineDeps:
    pan_detector: Any
    aadhaar_detector: Any
    ocr: Any
    doc_classifier: DocClassifier
    llm_cleaner: Optional[LLMKycCleaner]
    vlm_extractor: Optional[Any]
    policy_verifier: Optional[Any]
    config: PipelineConfig

_deps_var: ContextVar[PipelineDeps] = ContextVar("pipeline_deps")

def set_deps(deps) -> Token: return _deps_var.set(deps)
def get_deps() -> PipelineDeps: return _deps_var.get()
```

`invoke_graph()` sets the context var before invocation and resets it in a `finally` block. Nodes call `get_deps()` to access shared resources.

## Consequences

**Positive:**
- Zero extra arguments to node functions — clean `(state) -> state` signature
- Thread-safe by design (contextvars are per-task/per-thread)
- Testable — set mock deps before invoking individual nodes
- No global state — each graph invocation gets its own context

**Negative:**
- Implicit dependency — nodes' need for deps isn't visible in their signature
- `get_deps()` throws if called outside a graph invocation context
- Less familiar pattern than constructor injection

**Risks:**
- Context leaks if `finally` block doesn't run (mitigated by try/finally in invoke_graph)
