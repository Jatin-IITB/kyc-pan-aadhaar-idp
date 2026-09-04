"""Retrieval metrics and golden-set validation for the RAG evidence harness."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

GOLDEN_CATEGORIES = {"single", "multi", "negative", "near_miss"}
REPORT_FRACTION = 0.5


def split_for_question_id(
    question_id: str,
    report_fraction: float = REPORT_FRACTION,
) -> str:
    """Assign a stable fit/report split using only the authored question ID."""
    if not 0.0 < report_fraction < 1.0:
        raise ValueError("report_fraction must be between 0 and 1")
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "report" if bucket < report_fraction else "fit"


def load_golden(path: Path) -> List[Dict[str, Any]]:
    """Load and validate JSONL questions, adding their deterministic split."""
    records: List[Dict[str, Any]] = []
    seen_ids = set()

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: record must be a JSON object")

        question_id = record.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if question_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {question_id!r}")
        seen_ids.add(question_id)

        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{line_number}: question must be a non-empty string")

        category = record.get("category")
        if category not in GOLDEN_CATEGORIES:
            raise ValueError(
                f"{path}:{line_number}: category must be one of "
                f"{sorted(GOLDEN_CATEGORIES)}"
            )

        relevant = record.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or any(
            not isinstance(chunk_id, str) or not chunk_id for chunk_id in relevant
        ):
            raise ValueError(
                f"{path}:{line_number}: relevant_chunk_ids must be a list of strings"
            )
        if len(relevant) != len(set(relevant)):
            raise ValueError(
                f"{path}:{line_number}: relevant_chunk_ids contains duplicates"
            )

        expected_abstain = record.get("expected_abstain")
        if expected_abstain is not (category == "negative"):
            raise ValueError(
                f"{path}:{line_number}: expected_abstain must be true exactly "
                "for negative questions"
            )
        if expected_abstain and relevant:
            raise ValueError(
                f"{path}:{line_number}: negative questions cannot have relevant chunks"
            )
        if not expected_abstain and not relevant:
            raise ValueError(
                f"{path}:{line_number}: answerable questions need relevant chunks"
            )

        normalized = dict(record)
        normalized["split"] = split_for_question_id(question_id)
        records.append(normalized)

    if not records:
        raise ValueError(f"{path}: golden set is empty")
    return records


def validate_relevant_chunk_ids(
    golden: Iterable[Mapping[str, Any]],
    available_chunk_ids: Iterable[str],
) -> None:
    """Reject labels that do not map to a chunk derived from policy structure."""
    available = set(available_chunk_ids)
    missing = sorted(
        {
            chunk_id
            for question in golden
            for chunk_id in question["relevant_chunk_ids"]
            if chunk_id not in available
        }
    )
    if missing:
        raise ValueError(f"golden set references unknown policy chunks: {missing}")


def dataset_content_hash(golden: Iterable[Mapping[str, Any]]) -> str:
    """Hash authored records, excluding the derived split field."""
    canonical = []
    for record in sorted(golden, key=lambda item: str(item["id"])):
        canonical.append(
            {key: value for key, value in record.items() if key != "split"}
        )
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dcg(relevant: set[str], ranking: Sequence[str], k: int) -> float:
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranking[:k], 1)
        if chunk_id in relevant
    )


def evaluate_rankings(
    golden: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """Compute recall@1/5/10, MRR, nDCG@10, and negative abstention.

    Ranking metrics are macro-averaged over answerable questions. Negative
    questions are reported separately because they have no relevant item.
    """
    positive = [record for record in golden if record["relevant_chunk_ids"]]
    negative = [record for record in golden if record["expected_abstain"]]

    recalls = {1: [], 5: [], 10: []}
    reciprocal_ranks: List[float] = []
    ndcgs: List[float] = []

    for record in positive:
        relevant = set(record["relevant_chunk_ids"])
        ranking = list(rankings.get(str(record["id"]), []))
        for k in recalls:
            recalls[k].append(len(relevant.intersection(ranking[:k])) / len(relevant))

        first_relevant = next(
            (rank for rank, chunk_id in enumerate(ranking, 1) if chunk_id in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first_relevant if first_relevant else 0.0)

        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(relevant), 10) + 1)
        )
        ndcgs.append(_dcg(relevant, ranking, 10) / ideal if ideal else 0.0)

    abstentions = sum(
        not rankings.get(str(record["id"]), [])
        for record in negative
    )

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "n_questions": len(golden),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "recall_at_1": mean(recalls[1]),
        "recall_at_5": mean(recalls[5]),
        "recall_at_10": mean(recalls[10]),
        "mrr": mean(reciprocal_ranks),
        "ndcg_at_10": mean(ndcgs),
        "negative_abstention_rate": (
            round(abstentions / len(negative), 4) if negative else None
        ),
    }
