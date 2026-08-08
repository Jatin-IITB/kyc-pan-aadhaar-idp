from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Optional

from services.doc_classifier.classifier import DocClassifier
from services.extraction.llm_cleaner import LLMKycCleaner
from services.pipeline import PipelineConfig


@dataclass
class PipelineDeps:
    pan_detector: Any
    aadhaar_detector: Any
    ocr: Any
    doc_classifier: DocClassifier
    llm_cleaner: Optional[LLMKycCleaner]
    vlm_extractor: Optional[Any] = None
    policy_verifier: Optional[Any] = None
    rotation_classifier: Optional[Any] = None
    config: PipelineConfig = PipelineConfig()


_deps_var: contextvars.ContextVar[PipelineDeps] = contextvars.ContextVar("pipeline_deps")


def set_deps(deps: PipelineDeps) -> contextvars.Token:
    return _deps_var.set(deps)


def get_deps() -> PipelineDeps:
    return _deps_var.get()
