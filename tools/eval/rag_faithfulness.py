"""Citation-faithfulness evaluation for PolicyVerifier verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from services.rag.policy_verifier import PolicyVerifier
from tools.eval.rag_eval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    build_local_stack,
)

VALID_SUPPORT_LABELS = {"SUPPORTED", "UNSUPPORTED", "UNVERIFIABLE"}


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    seen = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"{path}:{line_number}: case id must be unique and non-empty")
        if not isinstance(record.get("doc_type"), str):
            raise ValueError(f"{path}:{line_number}: doc_type must be a string")
        if not isinstance(record.get("flat_fields"), dict):
            raise ValueError(f"{path}:{line_number}: flat_fields must be an object")
        seen.add(case_id)
        cases.append(record)
    if not cases:
        raise ValueError(f"{path}: citation case set is empty")
    return cases


def verdict_id(case_id: str, check_index: int, requirement: str) -> str:
    digest = hashlib.sha256(requirement.encode("utf-8")).hexdigest()[:12]
    return f"{case_id}:{check_index}:{digest}"


def verdict_content_hash(record: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {
            "case_data": record["case_data"],
            "requirement": record["requirement"],
            "status": record["status"],
            "citation": record["citation"],
            "explanation": record["explanation"],
        }
    )


def _resolve_citation(
    citation: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    candidates = [
        chunk
        for chunk in chunks
        if chunk["source_file"] == citation.get("source")
        and chunk["section_header"] == citation.get("section")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    excerpt = " ".join(str(citation.get("text", "")).split()).lower()
    if excerpt:
        for chunk in candidates:
            if excerpt in " ".join(str(chunk["text"]).split()).lower():
                return chunk
    return None


def _excerpt_is_in_chunk(
    citation: Mapping[str, Any],
    chunk: Optional[Mapping[str, Any]],
) -> bool:
    if not chunk:
        return False
    excerpt = " ".join(str(citation.get("text", "")).split()).lower()
    full_text = " ".join(str(chunk["text"]).split()).lower()
    return bool(excerpt) and excerpt in full_text


class CitationSupportJudge:
    """Deterministic Ollama judge using an explicit entailment rubric."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def judge(
        self,
        case_data: Mapping[str, Any],
        verdict: Mapping[str, Any],
        cited_chunk: Optional[Mapping[str, Any]],
    ) -> Dict[str, str]:
        if cited_chunk is None:
            return {
                "support": "UNVERIFIABLE",
                "reason": "The cited source and section do not resolve to an indexed chunk.",
            }

        system = (
            "You audit whether a regulatory citation supports a KYC verdict. "
            "Judge citation entailment only; do not reward plausibility or outside "
            "knowledge. SUPPORTED means the cited policy text, together with the "
            "stated case fields, directly supports both the verdict status and its "
            "explanation. UNSUPPORTED means the text is irrelevant, contradicts the "
            "verdict, or omits a premise needed for it. UNVERIFIABLE means the "
            "citation text is empty or malformed. Return only JSON with keys "
            '"support" (SUPPORTED, UNSUPPORTED, or UNVERIFIABLE) and "reason".'
        )
        user = json.dumps(
            {
                "case_data": case_data,
                "requirement": verdict["requirement"],
                "verdict_status": verdict["status"],
                "verdict_explanation": verdict["explanation"],
                "cited_source": verdict["citation"],
                "full_cited_policy_chunk": cited_chunk["text"],
            },
            ensure_ascii=False,
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.0, "seed": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"citation judge request failed: {exc}") from exc

        content = (body.get("message") or {}).get("content")
        parsed = json.loads(content)
        support = parsed.get("support")
        if support not in VALID_SUPPORT_LABELS:
            raise ValueError(f"citation judge returned invalid support label: {support!r}")
        return {
            "support": support,
            "reason": str(parsed.get("reason", "")),
        }


def load_human_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    labels = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if not isinstance(record.get("supported"), bool):
            raise ValueError(f"{path}:{line_number}: supported must be boolean")
        labels[str(record["id"])] = record
    return labels


def _select_hand_sample(verdicts: Iterable[Mapping[str, Any]], n: int = 20) -> List[str]:
    ids = [str(verdict["id"]) for verdict in verdicts]
    return sorted(ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:n]


def run_faithfulness(
    policies_dir: Path,
    cases_path: Path,
    human_labels_path: Path,
    ollama_url: str = "http://localhost:11434",
    verifier_model: str = "qwen3:8b",
    judge_model: str = "qwen3:8b",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    max_verdicts: Optional[int] = None,
) -> Dict[str, Any]:
    retriever, reranker, chunks = build_local_stack(
        policies_dir,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
    verifier = PolicyVerifier(
        retriever=retriever,
        reranker=reranker,
        llm_url=ollama_url,
        llm_model=verifier_model,
        timeout_s=120.0,
    )
    support_judge = CitationSupportJudge(
        base_url=ollama_url,
        model=judge_model,
    )
    cases = load_cases(cases_path)

    verdicts = []
    stop = False
    for case_data in cases:
        result = verifier.verify(case_data)
        for check_index, check in enumerate(result["checks"]):
            record = {
                "id": verdict_id(
                    str(case_data["id"]),
                    check_index,
                    str(check["requirement"]),
                ),
                "case_data": case_data,
                **check,
            }
            cited_chunk = _resolve_citation(check["citation"], chunks)
            record["resolved_chunk_id"] = (
                cited_chunk["chunk_id"] if cited_chunk else None
            )
            record["excerpt_in_chunk"] = _excerpt_is_in_chunk(
                check["citation"],
                cited_chunk,
            )
            record["judge"] = support_judge.judge(case_data, check, cited_chunk)
            record["verdict_sha256"] = verdict_content_hash(record)
            verdicts.append(record)
            print(
                f'  {record["id"]}: {record["judge"]["support"]}',
                flush=True,
            )
            if max_verdicts is not None and len(verdicts) >= max_verdicts:
                stop = True
                break
        if stop:
            break

    labels = load_human_labels(human_labels_path)
    hand_sample_ids = set(_select_hand_sample(verdicts))
    matched_labels = []
    stale_labels = []
    for verdict in verdicts:
        label = labels.get(verdict["id"])
        if not label or verdict["id"] not in hand_sample_ids:
            continue
        if label.get("verdict_sha256") != verdict["verdict_sha256"]:
            stale_labels.append(verdict["id"])
            continue
        matched_labels.append(
            (
                verdict["judge"]["support"] == "SUPPORTED",
                label["supported"],
            )
        )

    supported = sum(
        verdict["judge"]["support"] == "SUPPORTED" for verdict in verdicts
    )
    resolved = sum(verdict["resolved_chunk_id"] is not None for verdict in verdicts)
    exact_excerpts = sum(verdict["excerpt_in_chunk"] for verdict in verdicts)
    agreement = (
        sum(judge == human for judge, human in matched_labels) / len(matched_labels)
        if matched_labels
        else None
    )
    return {
        "cases_path": str(cases_path),
        "cases_sha256": _canonical_hash(
            {"cases": sorted(cases, key=lambda case: case["id"])}
        ),
        "models": {
            "verifier": verifier_model,
            "citation_judge": judge_model,
            "embedding": embedding_model,
            "reranker": reranker_model,
        },
        "n_verdicts": len(verdicts),
        "citation_support_rate": round(supported / len(verdicts), 4),
        "citation_resolution_rate": round(resolved / len(verdicts), 4),
        "cited_excerpt_in_chunk_rate": round(exact_excerpts / len(verdicts), 4),
        "judge_agreement": round(agreement, 4) if agreement is not None else None,
        "n_human_labels": len(matched_labels),
        "stale_human_label_ids": stale_labels,
        "hand_sample_ids": sorted(hand_sample_ids),
        "verdicts": verdicts,
    }


def write_hand_review_template(results: Mapping[str, Any], path: Path) -> None:
    selected = set(results["hand_sample_ids"])
    lines = []
    for verdict in results["verdicts"]:
        if verdict["id"] not in selected:
            continue
        lines.append(
            json.dumps(
                {
                    "id": verdict["id"],
                    "verdict_sha256": verdict["verdict_sha256"],
                    "supported": None,
                    "notes": "",
                },
                ensure_ascii=False,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG citation faithfulness")
    parser.add_argument("--policies-dir", type=Path, default=Path("config/policies"))
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/rag_eval/citation_cases.jsonl"),
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        default=Path("data/rag_eval/citation_human_labels.jsonl"),
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--verifier-model", default="qwen3:8b")
    parser.add_argument("--judge-model", default="qwen3:8b")
    parser.add_argument("--max-verdicts", type=int)
    parser.add_argument("--out", type=Path, default=Path("eval/rag_faithfulness.json"))
    parser.add_argument(
        "--hand-review-template",
        type=Path,
        default=Path("eval/citation_hand_review.jsonl"),
    )
    args = parser.parse_args()

    results = run_faithfulness(
        args.policies_dir,
        args.cases,
        args.human_labels,
        ollama_url=args.ollama_url,
        verifier_model=args.verifier_model,
        judge_model=args.judge_model,
        max_verdicts=args.max_verdicts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_hand_review_template(results, args.hand_review_template)
    print(f"wrote {args.out}")
    print(f"wrote {args.hand_review_template}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
