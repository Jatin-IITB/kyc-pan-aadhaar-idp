from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


class VLMExtractorError(RuntimeError):
    pass


VLM_PROMPTS: Dict[str, str] = {
    "pan": (
        "Extract fields from this Indian PAN card image.\n"
        "Return ONLY a JSON object with these exact keys: pan_number, name, father_name, date_of_birth\n"
        "For each key, provide the value as a string exactly as it appears on the card.\n"
        "If a field is not visible or unreadable, use empty string \"\".\n"
        "Dates must be in DD/MM/YYYY format.\n"
        "PAN number format: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F).\n"
        "Return ONLY the JSON object. No markdown. No commentary."
    ),
    "aadhaar": (
        "Extract fields from this Indian Aadhaar card image.\n"
        "Return ONLY a JSON object with these exact keys: aadhaar_number, name, date_of_birth, gender, address\n"
        "For each key, provide the value as a string exactly as it appears on the card.\n"
        "If a field is not visible or unreadable, use empty string \"\".\n"
        "Aadhaar number format: 4 digits space 4 digits space 4 digits (e.g., 1234 5678 9012).\n"
        "Dates must be in DD/MM/YYYY format.\n"
        "Gender must be one of: Male, Female, Other.\n"
        "Return ONLY the JSON object. No markdown. No commentary."
    ),
    "passport": (
        "Extract fields from this Indian passport image.\n"
        "Return ONLY a JSON object with these exact keys: passport_number, surname, given_names, "
        "nationality, date_of_birth, date_of_issue, date_of_expiry, place_of_birth, sex\n"
        "If a field is not visible or unreadable, use empty string \"\".\n"
        "Dates must be in DD/MM/YYYY format.\n"
        "Return ONLY the JSON object. No markdown. No commentary."
    ),
    "driving_license": (
        "Extract fields from this Indian driving license image.\n"
        "Return ONLY a JSON object with these exact keys: dl_number, name, date_of_birth, "
        "blood_group, address, date_of_issue, date_of_expiry\n"
        "If a field is not visible or unreadable, use empty string \"\".\n"
        "Dates must be in DD/MM/YYYY format.\n"
        "Return ONLY the JSON object. No markdown. No commentary."
    ),
    "voter_id": (
        "Extract fields from this Indian Voter ID (EPIC) card image.\n"
        "Return ONLY a JSON object with these exact keys: epic_number, name, "
        "father_name, date_of_birth, gender, address\n"
        "If a field is not visible or unreadable, use empty string \"\".\n"
        "Return ONLY the JSON object. No markdown. No commentary."
    ),
}

DEFAULT_PROMPT = (
    "Extract all text fields from this identity document image.\n"
    "Return ONLY a JSON object where keys are field names and values are the text content.\n"
    "If a field is not visible or unreadable, use empty string \"\".\n"
    "Return ONLY the JSON object. No markdown. No commentary."
)


@dataclass(frozen=True)
class VLMConfig:
    base_url: str = (os.getenv("KYC_OLLAMA_URL") or "http://host.docker.internal:11434").strip()
    model: str = (os.getenv("KYC_VLM_MODEL") or "llama3.2-vision:11b").strip()
    timeout_s: float = float((os.getenv("KYC_VLM_TIMEOUT_S") or "30").strip() or "30")


class VLMExtractor:
    """Ollama multimodal VLM wrapper for structured field extraction from document images."""

    def __init__(self, config: Optional[VLMConfig] = None) -> None:
        self.config = config or VLMConfig()

    def extract_fields(self, image_bgr: np.ndarray, doc_type: str) -> Dict[str, Dict[str, Any]]:
        b64_image = self._encode_image(image_bgr)
        prompt = VLM_PROMPTS.get(doc_type, DEFAULT_PROMPT)

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }

        url = self.config.base_url.rstrip("/") + "/api/generate"
        parsed = self._post_json(url, payload)

        raw_response = parsed.get("response", "")
        if not isinstance(raw_response, str) or not raw_response.strip():
            return {}

        try:
            fields = json.loads(raw_response)
        except json.JSONDecodeError:
            return {}

        if not isinstance(fields, dict):
            return {}

        extraction: Dict[str, Dict[str, Any]] = {}
        for key, value in fields.items():
            extraction[key] = {
                "value": str(value) if value else "",
                "det_conf": 0.0,
                "ocr_conf": 0.85,
                "bbox": None,
            }

        return extraction

    def _encode_image(self, image_bgr: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise VLMExtractorError("Failed to encode image as JPEG")
        return base64.b64encode(buffer).decode("ascii")

    def _post_json(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as r:
                body = r.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            raise VLMExtractorError(f"Ollama VLM request failed: {e}") from e

        try:
            result = json.loads(body)
        except json.JSONDecodeError as e:
            raise VLMExtractorError(f"VLM response not valid JSON: {e}") from e

        if not isinstance(result, dict):
            raise VLMExtractorError("VLM response was not a JSON object")

        err = result.get("error")
        if isinstance(err, str) and err.strip():
            raise VLMExtractorError(f"Ollama error: {err.strip()}")

        return result
