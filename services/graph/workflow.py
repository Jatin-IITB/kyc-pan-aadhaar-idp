from __future__ import annotations

from langgraph.graph import END, StateGraph

from services.graph.deps import PipelineDeps, set_deps
from services.graph.nodes.audit_commit import audit_commit_node
from services.graph.nodes.classify import classify_node
from services.graph.nodes.cross_doc import cross_doc_node
from services.graph.nodes.decide import decide_node
from services.graph.nodes.ensemble import ensemble_node
from services.graph.nodes.extract_vlm import extract_vlm_node
from services.graph.nodes.extract_yolo import extract_yolo_node
from services.graph.nodes.forensics import forensics_node
from services.graph.nodes.ingest import ingest_node
from services.graph.nodes.llm_rescue import llm_rescue_node
from services.graph.nodes.policy_verify import policy_verify_node
from services.graph.nodes.quality_gate import quality_gate_node
from services.graph.nodes.validate import validate_node
from services.graph.state import CaseState


def _quality_router(state: CaseState) -> str:
    if not state.get("quality_passed", True):
        return "reject"
    return "continue"


def _yolo_confidence_router(state: CaseState) -> str:
    conf = state.get("yolo_confidence", 0.0)
    if conf < 0.60:
        return "vlm"
    return "ensemble"


def _cross_doc_router(state: CaseState) -> str:
    if len(state.get("packet_documents", [])) >= 2:
        return "cross_doc"
    return "skip"


def build_kyc_graph(pipeline_deps: PipelineDeps) -> StateGraph:
    """Build and compile the KYC processing graph.

    Topology:
      ingest -> quality_gate -> [reject -> decide | continue -> classify]
      classify -> extract_yolo -> ensemble -> validate -> policy_verify
      classify -> forensics (parallel) -> decide
      policy_verify -> [cross_doc | llm_rescue]
      cross_doc -> llm_rescue
      llm_rescue -> decide -> audit_commit -> END
    """
    graph = StateGraph(CaseState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("classify", classify_node)
    graph.add_node("extract_yolo", extract_yolo_node)
    graph.add_node("extract_vlm", extract_vlm_node)
    graph.add_node("ensemble", ensemble_node)
    graph.add_node("validate", validate_node)
    graph.add_node("policy_verify", policy_verify_node)
    graph.add_node("cross_doc", cross_doc_node)
    graph.add_node("forensics", forensics_node)
    graph.add_node("llm_rescue", llm_rescue_node)
    graph.add_node("decide", decide_node)
    graph.add_node("audit_commit", audit_commit_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "quality_gate")

    graph.add_conditional_edges(
        "quality_gate",
        _quality_router,
        {"reject": "decide", "continue": "classify"},
    )

    graph.add_edge("classify", "extract_yolo")
    graph.add_edge("classify", "forensics")

    graph.add_conditional_edges(
        "extract_yolo",
        _yolo_confidence_router,
        {"vlm": "extract_vlm", "ensemble": "ensemble"},
    )
    graph.add_edge("extract_vlm", "ensemble")
    graph.add_edge("ensemble", "validate")
    graph.add_edge("validate", "policy_verify")

    graph.add_conditional_edges(
        "policy_verify",
        _cross_doc_router,
        {"cross_doc": "cross_doc", "skip": "llm_rescue"},
    )
    graph.add_edge("cross_doc", "llm_rescue")

    graph.add_edge("forensics", "decide")
    graph.add_edge("llm_rescue", "decide")

    graph.add_edge("decide", "audit_commit")
    graph.add_edge("audit_commit", END)

    return graph.compile()


def invoke_graph(compiled_graph, pipeline_deps: PipelineDeps, state: CaseState) -> CaseState:
    token = set_deps(pipeline_deps)
    try:
        return compiled_graph.invoke(state)
    finally:
        from services.graph.deps import _deps_var
        _deps_var.reset(token)
