import json

from services.rag.policy_verifier import PolicyVerifier
from tools.eval.rag_faithfulness import (
    CitationSupportJudge,
    _excerpt_is_in_chunk,
    _resolve_citation,
    load_cases,
    verdict_content_hash,
    verdict_id,
)


def test_policy_verifier_default_dependencies_are_lazy(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    verifier = PolicyVerifier(llm_url="http://localhost:11434", llm_model="test")
    assert verifier.retriever.qdrant_url == "http://localhost:6333"
    assert verifier.retriever._encoder is None
    assert verifier.reranker._model is None


def test_citation_resolution_requires_an_indexed_source_and_section():
    chunks = [
        {
            "chunk_id": "policy.md::Section 1::0",
            "source_file": "policy.md",
            "section_header": "Section 1",
            "text": "PAN is required.",
        }
    ]
    citation = {
        "source": "policy.md",
        "section": "Section 1",
        "text": "PAN is required.",
    }
    resolved = _resolve_citation(citation, chunks)
    assert resolved["chunk_id"] == "policy.md::Section 1::0"
    assert _excerpt_is_in_chunk(citation, resolved) is True

    invented = {**citation, "section": "Invented"}
    assert _resolve_citation(invented, chunks) is None


def test_missing_citation_is_unverifiable_without_calling_ollama():
    judge = CitationSupportJudge(base_url="http://not-called.invalid")
    result = judge.judge(
        {"doc_type": "pan", "flat_fields": {}},
        {
            "requirement": "Is PAN required?",
            "status": "PASS",
            "explanation": "Yes.",
            "citation": {},
        },
        cited_chunk=None,
    )
    assert result["support"] == "UNVERIFIABLE"


def test_case_and_verdict_ids_are_reproducible(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    case = {"id": "case-1", "doc_type": "pan", "flat_fields": {"pan_number": "X"}}
    cases_path.write_text(json.dumps(case) + "\n")
    assert load_cases(cases_path) == [case]

    assert verdict_id("case-1", 0, "Question?") == verdict_id(
        "case-1",
        0,
        "Question?",
    )
    verdict = {
        "case_data": case,
        "requirement": "Question?",
        "status": "PASS",
        "citation": {"source": "x", "section": "y", "text": "z"},
        "explanation": "Because.",
    }
    assert verdict_content_hash(verdict) == verdict_content_hash(dict(verdict))


def test_policy_verifier_exposes_decision_and_audit_adapter_keys(monkeypatch):
    class Retriever:
        def retrieve(self, query, top_k):
            return [{"text": "PAN is required."}]

    class Reranker:
        def rerank(self, query, chunks, top_k):
            return chunks

    verifier = PolicyVerifier(retriever=Retriever(), reranker=Reranker())
    monkeypatch.setattr(verifier, "_generate_queries", lambda *_: ["Is PAN required?"])
    monkeypatch.setattr(
        verifier,
        "_judge_requirement",
        lambda *_: {
            "requirement": "Is PAN required?",
            "status": "PASS",
            "citation": {
                "source": "policy.md",
                "section": "PAN",
                "text": "PAN is required.",
            },
            "explanation": "The policy requires PAN.",
        },
    )

    result = verifier.verify({"doc_type": "pan", "flat_fields": {}})
    assert result["overall_status"] == "COMPLIANT"
    assert result["compliant"] is True
    assert result["citations"] == [result["checks"][0]["citation"]]


def test_policy_retrieval_failure_requires_review_and_fails_closed(monkeypatch):
    class FailingRetriever:
        def retrieve(self, query, top_k):
            raise RuntimeError("qdrant unavailable")

    verifier = PolicyVerifier(
        retriever=FailingRetriever(),
        reranker=object(),
    )
    monkeypatch.setattr(verifier, "_generate_queries", lambda *_: ["Requirement"])

    result = verifier.verify({"doc_type": "pan", "flat_fields": {"name": "A"}})
    assert result["overall_status"] == "REQUIRES_REVIEW"
    assert result["compliant"] is False
    assert result["citations"] == []
    assert result["checks"][0]["status"] == "INSUFFICIENT_DATA"
