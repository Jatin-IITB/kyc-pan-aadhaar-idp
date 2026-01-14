# services/extraction/llm_cleaner.py
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_OLLAMA_BASE_URL = (os.getenv("KYC_OLLAMA_URL") or "http://host.docker.internal:11434").strip()
DEFAULT_OLLAMA_MODEL = (os.getenv("KYC_OLLAMA_MODEL") or "llama3.2:3b").strip()
DEFAULT_TIMEOUT_S = float((os.getenv("KYC_OLLAMA_TIMEOUT_S") or "20").strip() or "20")

OLLAMA_GENERATE_PATH = "/api/generate"
OLLAMA_FORMAT_JSON = "json"
TEMPERATURE = 0.0


class LLMCleanerError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMCleanerConfig:
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S


class LLMKycCleaner:
    """
    Thin Ollama wrapper for OCR-noise cleanup only.
    Contract:
      - Input: flat dict {key: raw_string}
      - Output: dict {same_key: cleaned_string}; no new keys
      - Raises LLMCleanerError on any Ollama-side error
    """

    def __init__(self, config: Optional[LLMCleanerConfig] = None) -> None:
        self.config = config or LLMCleanerConfig()

    def clean_fields(
        self,
        *,
        doc_type: str,
        fields: Dict[str, Any],
        failure_reason: Optional[str],
    ) -> Dict[str, Any]:
        if not isinstance(fields, dict) or not fields:
            return {}

        # Schema-law keys are whatever the pipeline passes (already normalized/mapped).
        allowed_keys = list(fields.keys())

        payload = {
            "model": self.config.model,
            "prompt": self._build_prompt(
                doc_type=doc_type,
                allowed_keys=allowed_keys,
                fields=fields,
                failure_reason=failure_reason,
            ),
            "stream": False,
            "format": OLLAMA_FORMAT_JSON,
            "options": {"temperature": TEMPERATURE},
        }

        resp = self._post_json(self._build_url(OLLAMA_GENERATE_PATH), payload, timeout_s=self.config.timeout_s)

        # Ollama success must contain response + done=true; otherwise treat as error.
        err = resp.get("error")
        if isinstance(err, str) and err.strip():
            raise LLMCleanerError(f"Ollama error: {err.strip()}")

        if resp.get("done") is not True:
            raise LLMCleanerError(f"Ollama generation not done: done={resp.get('done')}")

        raw = resp.get("response")
        if not isinstance(raw, str) or not raw.strip():
            raise LLMCleanerError("Ollama returned empty 'response'")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMCleanerError(f"Ollama response was not valid JSON: {e}") from e

        if not isinstance(parsed, dict):
            raise LLMCleanerError("Ollama JSON was not an object")

        # Hard schema lock: only return keys that already exist in input.
        out: Dict[str, Any] = {}
        for k in allowed_keys:
            if k in parsed:
                out[k] = parsed.get(k)
        return out

    def _build_url(self, path: str) -> str:
        base = (self.config.base_url or "").strip()
        if not base:
            raise LLMCleanerError("Missing Ollama base_url (KYC_OLLAMA_URL)")
        return base.rstrip("/") + path

    def _post_json(self, url: str, payload: Dict[str, Any], *, timeout_s: float) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                body = r.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            raise LLMCleanerError(f"Ollama request failed: {e}") from e

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as e:
            raise LLMCleanerError(f"Ollama HTTP 200 but body was not JSON: {e}") from e

        if not isinstance(parsed, dict):
            raise LLMCleanerError("Ollama HTTP 200 but JSON was not an object")

        # If Ollama returned an error object, surface it immediately.
        err = parsed.get("error")
        if isinstance(err, str) and err.strip():
            raise LLMCleanerError(f"Ollama error: {err.strip()}")

        # Guard against “optimistic empty dict” failure modes.
        if "response" not in parsed and "done" not in parsed:
            raise LLMCleanerError(f"Ollama unexpected response keys: {list(parsed.keys())}")

        return parsed

    def _build_prompt(
        self,
        *,
        doc_type: str,
        allowed_keys: list[str],
        fields: Dict[str, Any],
        failure_reason: Optional[str],
    ) -> str:
        keys_json = json.dumps(allowed_keys, ensure_ascii=False)
        fields_json = json.dumps(fields, ensure_ascii=False)

        reason_line = ""
        if isinstance(failure_reason, str) and failure_reason.strip():
            reason_line = f"Schema failure reason: {failure_reason.strip()}\n"

        system = (
            "You are a strict KYC OCR-cleaning function for Indian IDs (PAN/Aadhaar).\n"
            "Task: fix OCR noise only.\n"
            "You MUST NOT invent data. If unsure, keep the original value unchanged.\n"
            "You MUST NOT add or remove keys; output exactly and only the allowed keys.\n"
            "Dates: output must follow DD/MM/YYYY OR YYYY only (no other formats).\n"
            "For date_of_birth: do not change the year unless the original clearly contains the corrected year.\n"
            "For pan_number and aadhaar_number: do not change digits/letters except confusable OCR swaps.\n"
            "For names: you may replace '_' with space and collapse repeated spaces.\n"
            "Return ONLY a JSON object. No markdown. No commentary.\n"
        )

        # Few-shot examples for underscore and date confusables
        ex_user_1 = json.dumps(
            {"name": "RAHUL_KUMAR", "date_of_birth": "I0/0N/1995"},
            ensure_ascii=False,
        )
        ex_assistant_1 = json.dumps(
            {"name": "RAHUL KUMAR", "date_of_birth": "10/01/1995"},
            ensure_ascii=False,
        )

        ex_user_2 = json.dumps(
            {"pan_number": "ABCDEI234F", "date_of_birth": "12/05/____"},
            ensure_ascii=False,
        )
        ex_assistant_2 = json.dumps(
            {"pan_number": "ABCDE1234F", "date_of_birth": "12/05/____"},
            ensure_ascii=False,
        )

        user = (
            f"{reason_line}"
            f"Document type: {doc_type}\n"
            f"Allowed keys (MUST match exactly): {keys_json}\n"
            f"Input JSON:\n{fields_json}\n"
            "Return ONLY the cleaned JSON object now."
        )

        # Llama 3.1/3.2 chat template
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{ex_user_1}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"{ex_assistant_1}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{ex_user_2}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"{ex_assistant_2}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
