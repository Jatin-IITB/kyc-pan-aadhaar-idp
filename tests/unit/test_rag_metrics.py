import json
from pathlib import Path

import pytest

from services.rag.indexer import PolicyIndexer, canonical_chunk_id
from services.rag.retriever import HybridRetriever
from tools.eval.metrics import check_gates
from tools.eval.rag_eval import CONFIGURATIONS, _retrieve
from tools.eval.rag_metrics import (
    dataset_content_hash,
    evaluate_rankings,
    load_golden,
    split_for_question_id,
    validate_relevant_chunk_ids,
)


def test_ranking_metrics_cover_multi_relevance_and_abstention():
    golden = [
        {
            "id": "q1",
            "relevant_chunk_ids": ["a", "b"],
            "expected_abstain": False,
        },
        {
            "id": "q2",
            "relevant_chunk_ids": ["c"],
            "expected_abstain": False,
        },
        {
            "id": "q3",
            "relevant_chunk_ids": [],
            "expected_abstain": True,
        },
    ]
    metrics = evaluate_rankings(
        golden,
        {
            "q1": ["a", "b"],
            "q2": ["x", "c"],
            "q3": [],
        },
    )

    assert metrics["recall_at_1"] == 0.25
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr"] == 0.75
    assert metrics["ndcg_at_10"] == 0.8155
    assert metrics["negative_abstention_rate"] == 1.0


def test_split_is_stable_and_depends_only_on_question_id():
    assert split_for_question_id("rag-q001") == split_for_question_id("rag-q001")
    assignments = {
        split_for_question_id(f"question-{index}") for index in range(100)
    }
    assert assignments == {"fit", "report"}


def test_load_golden_validates_negative_contract_and_hash(tmp_path):
    path = tmp_path / "golden.jsonl"
    records = [
        {
            "id": "positive",
            "category": "single",
            "question": "Supported?",
            "relevant_chunk_ids": ["source.md::Section::0"],
            "expected_abstain": False,
        },
        {
            "id": "negative",
            "category": "negative",
            "question": "Unsupported?",
            "relevant_chunk_ids": [],
            "expected_abstain": True,
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))

    loaded = load_golden(path)
    assert {record["split"] for record in loaded}.issubset({"fit", "report"})
    assert dataset_content_hash(loaded) == dataset_content_hash(
        list(reversed(loaded))
    )

    records[1]["relevant_chunk_ids"] = ["invented"]
    path.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(ValueError, match="negative questions cannot"):
        load_golden(path)


def test_committed_golden_set_has_authored_coverage_and_valid_chunk_ids():
    golden = load_golden(Path("data/rag_eval/golden.jsonl"))
    assert 60 <= len(golden) <= 80
    assert {record["category"] for record in golden} == {
        "single",
        "multi",
        "negative",
        "near_miss",
    }
    assert {record["split"] for record in golden} == {"fit", "report"}

    indexer = PolicyIndexer(qdrant_url="unused")
    chunks = indexer.load_policy_chunks(Path("config/policies"))
    validate_relevant_chunk_ids(
        golden,
        [chunk["chunk_id"] for chunk in chunks],
    )


def test_structural_chunk_ids_are_stable():
    indexer = PolicyIndexer(qdrant_url="unused")
    text = "# Policy\n\n## Rule 1\nAlpha\n\n## Rule 2\nBeta"
    first = indexer._chunk_markdown(text, source="policy.md")
    second = indexer._chunk_markdown(text, source="policy.md")

    assert [chunk["chunk_id"] for chunk in first] == [
        chunk["chunk_id"] for chunk in second
    ]
    assert first[1]["chunk_id"] == canonical_chunk_id(
        "policy.md",
        "Rule 1",
        0,
    )


def test_rrf_rewards_documents_returned_by_both_rankers():
    dense = [
        {"id": "dense-only", "chunk_id": "d"},
        {"id": "shared", "chunk_id": "s"},
    ]
    sparse = [
        {"id": "shared", "chunk_id": "s"},
        {"id": "sparse-only", "chunk_id": "b"},
    ]
    fused = HybridRetriever._reciprocal_rank_fusion(dense, sparse)
    assert fused[0]["id"] == "shared"


def test_rag_gates_use_report_metrics_and_skip_an_unrun_tier():
    thresholds = {
        "rag": {
            "recall_at_5_min": 0.9,
            "mrr_min": 0.8,
            "citation_support_rate_min": 0.7,
        }
    }
    gates = check_gates(
        {"rag": {"recall_at_5": 0.95, "mrr": 0.85}},
        thresholds,
    )
    statuses = {result["gate"]: result["status"] for result in gates["results"]}
    assert statuses == {
        "rag.recall_at_5": "PASS",
        "rag.mrr": "PASS",
        "rag.citation_support_rate": "SKIPPED",
    }

    failed = check_gates(
        {"rag": {"recall_at_5": 0.89, "mrr": 0.85}},
        thresholds,
    )
    assert failed["passed"] is False


def test_ablation_configurations_call_distinct_retrieval_paths():
    class Retriever:
        def retrieve_dense(self, query, top_k):
            return [{"chunk_id": "dense", "text": query}]

        def retrieve_bm25(self, query, top_k):
            return [{"chunk_id": "bm25", "text": query}]

        def retrieve_rrf(self, query, top_k):
            return [{"chunk_id": "rrf", "text": query}]

    class Reranker:
        def rerank(self, query, chunks, top_k):
            return [{**chunk, "reranked": True} for chunk in chunks]

    retriever, reranker = Retriever(), Reranker()
    outputs = {
        configuration: _retrieve(
            configuration,
            "query",
            retriever,
            reranker,
            top_k=10,
        )
        for configuration in CONFIGURATIONS
    }

    assert outputs["dense"][0]["chunk_id"] == "dense"
    assert outputs["bm25"][0]["chunk_id"] == "bm25"
    assert outputs["rrf"][0]["chunk_id"] == "rrf"
    assert outputs["rrf_cross_encoder"][0] == {
        "chunk_id": "rrf",
        "text": "query",
        "reranked": True,
    }
    assert outputs["bm25_cross_encoder"][0]["chunk_id"] == "bm25"
