# services/rag/policy_verifier.py
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

from services.rag.reranker import CrossEncoderReranker
from services.rag.retriever import HybridRetriever

DEFAULT_OLLAMA_URL = (os.getenv("KYC_OLLAMA_URL") or "http://host.docker.internal:11434").strip()
DEFAULT_OLLAMA_MODEL = (os.getenv("KYC_OLLAMA_MODEL") or "llama3.2:3b").strip()
DEFAULT_TIMEOUT_S = float((os.getenv("KYC_OLLAMA_TIMEOUT_S") or "30").strip() or "30")

OLLAMA_CHAT_PATH = "/api/chat"

# Allowed per-check statuses returned by the LLM judge
VALID_STATUSES = {"PASS", "FAIL", "NOT_APPLICABLE", "INSUFFICIENT_DATA"}


class PolicyVerifierError(RuntimeError):
    pass


class PolicyVerifier:
    """RAG-based policy compliance verifier.

    Given extracted KYC fields and document type, this class:
      1. Generates verification queries from the case data.
      2. Retrieves relevant policy chunks via hybrid search.
      3. Reranks chunks for precision with a cross-encoder.
      4. Uses an Ollama LLM to judge each requirement against the
         retrieved policy text.
      5. Returns a structured compliance result with citations.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm_url: str = DEFAULT_OLLAMA_URL,
        llm_model: str = DEFAULT_OLLAMA_MODEL,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm_url = llm_url.rstrip("/")
        self.llm_model = llm_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify compliance of *case_data* against indexed policies.

        Args:
            case_data: Must contain at minimum ``doc_type`` (str) and
                ``flat_fields`` (dict of extracted field values).

        Returns:
            A dict with structure::

                {
                    "overall_status": "COMPLIANT" | "NON_COMPLIANT" | "REQUIRES_REVIEW",
                    "checks": [
                        {
                            "requirement": str,
                            "status": "PASS" | "FAIL" | "NOT_APPLICABLE" | "INSUFFICIENT_DATA",
                            "citation": {"source": str, "section": str, "text": str},
                            "explanation": str,
                        },
                        ...
                    ],
                }
        """
        doc_type = case_data.get("doc_type", "unknown")
        flat_fields = case_data.get("flat_fields", {})

        queries = self._generate_queries(doc_type, flat_fields)
        if not queries:
            return {
                "overall_status": "REQUIRES_REVIEW",
                "checks": [],
            }

        checks: List[Dict[str, Any]] = []
        for query in queries:
            retrieved = self.retriever.retrieve(query, top_k=10)
            reranked = self.reranker.rerank(query, retrieved, top_k=5)
            judgement = self._judge_requirement(query, reranked, doc_type, flat_fields)
            checks.append(judgement)

        overall = self._determine_overall_status(checks)

        return {
            "overall_status": overall,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    def _generate_queries(
        self, doc_type: str, flat_fields: Dict[str, Any]
    ) -> List[str]:
        """Generate verification queries from the document type and
        extracted fields."""
        queries: List[str] = []

        doc_label = doc_type.replace("_", " ").upper()

        # Mandatory document queries
        queries.append(
            f"What documents are mandatory for KYC verification under Indian regulations?"
        )
        queries.append(
            f"Is {doc_label} an acceptable Officially Valid Document (OVD) for KYC?"
        )

        # Field-specific checks
        if "pan_number" in flat_fields:
            queries.append(
                "Is PAN card mandatory for opening a bank account under RBI KYC norms?"
            )
        if "aadhaar_number" in flat_fields:
            queries.append(
                "What are the rules for Aadhaar-based e-KYC authentication?"
            )
            queries.append(
                "Is Aadhaar sufficient as standalone proof of identity and address?"
            )

        # Due diligence level
        queries.append(
            f"What level of due diligence is required for {doc_label} verification?"
        )

        return queries

    # ------------------------------------------------------------------
    # LLM judgement
    # ------------------------------------------------------------------

    def _judge_requirement(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        doc_type: str,
        flat_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LLM to judge a single requirement against retrieved
        policy chunks.  Returns a check dict with citation."""
        context = "\n\n".join(
            f"[Source: {c.get('source_file', '?')} | Section: {c.get('section_header', '?')}]\n{c['text']}"
            for c in chunks
        )

        fields_summary = json.dumps(flat_fields, ensure_ascii=False, default=str)

        messages = self._build_judge_messages(query, context, doc_type, fields_summary)

        try:
            response = self._call_ollama(messages)
            parsed = self._parse_judgement(response)
        except Exception as e:
            logger.warning("Policy judgement failed for query '%s': %s", query, e)
            parsed = {
                "status": "INSUFFICIENT_DATA",
                "explanation": f"LLM judgement failed: {e}",
                "cited_source": "",
                "cited_section": "",
                "cited_text": "",
            }

        best_chunk = chunks[0] if chunks else {}
        return {
            "requirement": query,
            "status": parsed.get("status", "INSUFFICIENT_DATA"),
            "citation": {
                "source": parsed.get("cited_source") or best_chunk.get("source_file", ""),
                "section": parsed.get("cited_section") or best_chunk.get("section_header", ""),
                "text": parsed.get("cited_text") or best_chunk.get("text", "")[:200],
            },
            "explanation": parsed.get("explanation", ""),
        }

    def _build_judge_messages(
        self,
        query: str,
        context: str,
        doc_type: str,
        fields_summary: str,
    ) -> List[Dict[str, str]]:
        system = (
            "You are an Indian KYC compliance officer.\n"
            "Given a regulatory question and policy context, determine whether "
            "the presented document/fields satisfy the requirement.\n"
            "Respond ONLY with a JSON object with these keys:\n"
            '  "status": one of PASS, FAIL, NOT_APPLICABLE, INSUFFICIENT_DATA\n'
            '  "explanation": brief reasoning (1-2 sentences)\n'
            '  "cited_source": the source file name of the most relevant policy\n'
            '  "cited_section": the section header of the most relevant policy\n'
            '  "cited_text": a short excerpt (under 100 words) from the cited policy\n'
            "No markdown. No commentary outside the JSON.\n"
        )

        user = (
            f"Document type: {doc_type}\n"
            f"Extracted fields: {fields_summary}\n\n"
            f"Regulatory question: {query}\n\n"
            f"Policy context:\n{context}\n\n"
            "Return ONLY the JSON object now."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat request to Ollama and return the response text."""
        url = f"{self.llm_url}{OLLAMA_CHAT_PATH}"
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as r:
                body = r.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            raise PolicyVerifierError(f"Ollama request failed: {e}") from e

        resp = json.loads(body)
        err = resp.get("error")
        if isinstance(err, str) and err.strip():
            raise PolicyVerifierError(f"Ollama error: {err.strip()}")

        message = resp.get("message", {})
        raw = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(raw, str) or not raw.strip():
            raise PolicyVerifierError("Ollama returned empty response")

        return raw

    def _parse_judgement(self, raw: str) -> Dict[str, Any]:
        """Parse the LLM JSON response, validating the status field."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise PolicyVerifierError(f"LLM response not valid JSON: {e}") from e

        if not isinstance(parsed, dict):
            raise PolicyVerifierError("LLM response was not a JSON object")

        status = parsed.get("status", "INSUFFICIENT_DATA")
        if status not in VALID_STATUSES:
            parsed["status"] = "INSUFFICIENT_DATA"

        return parsed

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_overall_status(checks: List[Dict[str, Any]]) -> str:
        """Derive the overall compliance status from individual checks.

        - Any FAIL -> NON_COMPLIANT
        - Any INSUFFICIENT_DATA (and no FAILs) -> REQUIRES_REVIEW
        - All PASS or NOT_APPLICABLE -> COMPLIANT
        """
        statuses = {c.get("status") for c in checks}

        if "FAIL" in statuses:
            return "NON_COMPLIANT"
        if "INSUFFICIENT_DATA" in statuses:
            return "REQUIRES_REVIEW"
        return "COMPLIANT"
