# apps/common/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _env(key: str) -> Optional[str]:
    v = os.getenv(key)
    return v.strip() if isinstance(v, str) and v.strip() else None


def _as_path(v: str) -> Path:
    return Path(v).expanduser().resolve()


def _as_paths(vs: List[str]) -> List[Path]:
    return [_as_path(v) for v in vs]


@dataclass(frozen=True)
class AppSettings:
    gateway_url: str
    eval_results_path: Path
    image_roots: List[Path]
    thresholds_path: Path


def load_settings(config_path: Optional[str] = None) -> AppSettings:
    """
    Resolution order (highest -> lowest):
      1) Explicit function argument
      2) KYC_CONFIG_PATH env var
      3) config/app.yaml
    Individual fields can be overridden via env vars:
      - KYC_GATEWAY_URL
      - KYC_EVAL_RESULTS_PATH
      - KYC_IMAGE_ROOTS (comma-separated)
      - KYC_THRESHOLDS_PATH
    """
    cfg_path = (
        Path(config_path)
        if config_path
        else Path(_env("KYC_CONFIG_PATH") or "config/app.yaml")
    )
    cfg = _read_yaml(cfg_path)

    gateway_url = _env("KYC_GATEWAY_URL") or cfg.get("gateway_url")
    eval_results_path = _env("KYC_EVAL_RESULTS_PATH") or cfg.get("eval_results_path")
    thresholds_path = _env("KYC_THRESHOLDS_PATH") or cfg.get("thresholds_path")

    image_roots_env = _env("KYC_IMAGE_ROOTS")
    if image_roots_env:
        image_roots_raw = [s.strip() for s in image_roots_env.split(",") if s.strip()]
    else:
        image_roots_raw = cfg.get("image_roots") or []

    missing = []
    if not gateway_url:
        missing.append("gateway_url / KYC_GATEWAY_URL")
    if not eval_results_path:
        missing.append("eval_results_path / KYC_EVAL_RESULTS_PATH")
    if not thresholds_path:
        missing.append("thresholds_path / KYC_THRESHOLDS_PATH")
    if not image_roots_raw:
        missing.append("image_roots / KYC_IMAGE_ROOTS")

    if missing:
        raise ValueError(
            "Missing required configuration: " + ", ".join(missing) +
            f". Config file used: {cfg_path}"
        )

    return AppSettings(
        gateway_url=str(gateway_url),
        eval_results_path=_as_path(str(eval_results_path)),
        image_roots=_as_paths([str(x) for x in image_roots_raw]),
        thresholds_path=_as_path(str(thresholds_path)),
    )
