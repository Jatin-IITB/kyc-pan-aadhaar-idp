"""Reproducible component ablation for the policy RAG pipeline.

The golden labels are authored from Markdown section boundaries. This runner
never uses retriever output to construct or modify relevance labels.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from services.rag.indexer import PolicyIndexer
from services.rag.reranker import CrossEncoderReranker
from services.rag.retriever import HybridRetriever
from tools.eval.metrics import check_gates
from tools.eval.rag_metrics import (
    dataset_content_hash,
    evaluate_rankings,
    load_golden,
    validate_relevant_chunk_ids,
)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SHIPPED_CONFIGURATION = "rrf_cross_encoder"
CONFIGURATIONS = (
    "dense",
    "bm25",
    "rrf",
    SHIPPED_CONFIGURATION,
    "bm25_cross_encoder",
)


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def build_local_stack(
    policies_dir: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
) -> Tuple[HybridRetriever, CrossEncoderReranker, List[Dict[str, Any]]]:
    """Index the real policy chunks into an ephemeral Qdrant collection."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(location=":memory:")
    encoder = SentenceTransformer(embedding_model)
    indexer = PolicyIndexer(
        qdrant_url=":memory:",
        embedding_model=embedding_model,
        encoder=encoder,
        qdrant_client=client,
    )
    chunks = indexer.load_policy_chunks(policies_dir)
    if not chunks:
        raise ValueError(f"no policy chunks found in {policies_dir}")
    indexer.index_policies(policies_dir)

    retriever = HybridRetriever(
        qdrant_url=":memory:",
        embedding_model=embedding_model,
        encoder=encoder,
        qdrant_client=client,
    )
    reranker = CrossEncoderReranker(reranker_model)
    return retriever, reranker, chunks


def _retrieve(
    configuration: str,
    query: str,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    top_k: int,
) -> List[Dict[str, Any]]:
    if configuration == "dense":
        return retriever.retrieve_dense(query, top_k=top_k)
    if configuration == "bm25":
        return retriever.retrieve_bm25(query, top_k=top_k)
    if configuration == "rrf":
        return retriever.retrieve_rrf(query, top_k=top_k)
    if configuration == "rrf_cross_encoder":
        candidates = retriever.retrieve_rrf(query, top_k=top_k)
        return reranker.rerank(query, candidates, top_k=top_k)
    if configuration == "bm25_cross_encoder":
        candidates = retriever.retrieve_bm25(query, top_k=top_k)
        return reranker.rerank(query, candidates, top_k=top_k)
    raise ValueError(f"unknown RAG configuration: {configuration}")


def evaluate_configuration(
    configuration: str,
    golden: Sequence[Mapping[str, Any]],
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Run one architecture variant over fixed questions and labels."""
    rankings: Dict[str, List[str]] = {}
    latencies_ms: List[float] = []

    for record in golden:
        started = time.perf_counter()
        chunks = _retrieve(
            configuration,
            str(record["question"]),
            retriever,
            reranker,
            top_k,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        rankings[str(record["id"])] = [
            str(chunk["chunk_id"]) for chunk in chunks
        ]

    split_metrics = {
        split: evaluate_rankings(
            [record for record in golden if record["split"] == split],
            rankings,
        )
        for split in ("fit", "report")
    }
    split_metrics["all"] = evaluate_rankings(golden, rankings)

    failures = []
    for record in golden:
        relevant = set(record["relevant_chunk_ids"])
        if not relevant:
            continue
        ranking = rankings[str(record["id"])]
        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(ranking, 1)
                if chunk_id in relevant
            ),
            None,
        )
        if first_rank != 1 or len(relevant.intersection(ranking[:10])) < len(relevant):
            failures.append(
                {
                    "id": record["id"],
                    "split": record["split"],
                    "category": record["category"],
                    "first_relevant_rank": first_rank,
                    "relevant_chunk_ids": list(record["relevant_chunk_ids"]),
                    "retrieved_chunk_ids": ranking,
                }
            )

    return {
        "metrics": split_metrics,
        "latency_ms": {
            "p50": round(_percentile(latencies_ms, 50), 2),
            "p95": round(_percentile(latencies_ms, 95), 2),
            "mean": round(statistics.fmean(latencies_ms), 2),
        },
        "rankings": rankings,
        "failures": failures,
    }


def run_ablation(
    policies_dir: Path,
    golden_path: Path,
    configurations: Sequence[str] = CONFIGURATIONS,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
) -> Dict[str, Any]:
    """Run all requested variants and return a serializable evidence record."""
    invalid = sorted(set(configurations).difference(CONFIGURATIONS))
    if invalid:
        raise ValueError(f"unknown RAG configurations: {invalid}")

    golden = load_golden(golden_path)
    retriever, reranker, chunks = build_local_stack(
        policies_dir,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    validate_relevant_chunk_ids(golden, chunk_ids)

    results = {}
    for configuration in configurations:
        print(f"  RAG ablation: {configuration}...", flush=True)
        results[configuration] = evaluate_configuration(
            configuration,
            golden,
            retriever,
            reranker,
        )

    split_counts = Counter(str(record["split"]) for record in golden)
    category_counts = Counter(str(record["category"]) for record in golden)
    return {
        "dataset": {
            "path": str(golden_path),
            "sha256": dataset_content_hash(golden),
            "n_questions": len(golden),
            "split_counts": dict(sorted(split_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "corpus": {
            "path": str(policies_dir),
            "n_chunks": len(chunks),
            "chunk_ids": chunk_ids,
        },
        "models": {
            "embedding": embedding_model,
            "reranker": reranker_model,
        },
        "shipped_configuration": SHIPPED_CONFIGURATION,
        "configurations": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Policy RAG component ablation")
    parser.add_argument("--policies-dir", type=Path, default=Path("config/policies"))
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("data/rag_eval/golden.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("eval/rag_metrics.json"))
    parser.add_argument(
        "--configuration",
        dest="configurations",
        action="append",
        choices=CONFIGURATIONS,
        help="run only this configuration; may be repeated",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="also run PolicyVerifier + qwen3 citation support judging",
    )
    parser.add_argument(
        "--citation-cases",
        type=Path,
        default=Path("data/rag_eval/citation_cases.jsonl"),
    )
    parser.add_argument(
        "--citation-human-labels",
        type=Path,
        default=Path("data/rag_eval/citation_human_labels.jsonl"),
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    results = run_ablation(
        args.policies_dir,
        args.golden,
        configurations=args.configurations or CONFIGURATIONS,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
    )

    shipped = results["configurations"].get(SHIPPED_CONFIGURATION)
    rag_gate_metrics = dict(shipped["metrics"]["report"]) if shipped else {}
    if args.faithfulness:
        from tools.eval.rag_faithfulness import run_faithfulness

        faithfulness = run_faithfulness(
            args.policies_dir,
            args.citation_cases,
            args.citation_human_labels,
            ollama_url=args.ollama_url,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
        )
        results["faithfulness"] = faithfulness
        rag_gate_metrics.update(
            {
                "citation_support_rate": faithfulness["citation_support_rate"],
                "judge_agreement": faithfulness["judge_agreement"],
            }
        )

    thresholds = yaml.safe_load(Path("config/eval_thresholds.yaml").read_text()) or {}
    gates = check_gates(
        {
            "rag": rag_gate_metrics or None,
        },
        thresholds,
    )
    results["gates"] = gates

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    for result in gates["results"]:
        if result["gate"].startswith("rag."):
            print(
                f'  {result["status"]:<8} {result["gate"]:<32} '
                f'limit={result["limit"]} actual={result["actual"]}'
            )

    return 0 if gates["passed"] or not args.check else 1


if __name__ == "__main__":
    sys.exit(main())
