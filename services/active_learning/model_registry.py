from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Track model versions with promote/rollback support."""

    def __init__(self, registry_path: str = "models/registry.json") -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._registry = self._load()

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {"models": {}, "active": {}}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._registry, indent=2, default=str))

    def register(
        self,
        model_name: str,
        version: str,
        weights_path: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        models = self._registry.setdefault("models", {})
        versions = models.setdefault(model_name, [])

        entry = {
            "version": version,
            "weights_path": weights_path,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "status": "registered",
        }

        existing = [v for v in versions if v["version"] == version]
        if existing:
            existing[0].update(entry)
        else:
            versions.append(entry)

        self._save()
        logger.info("Registered %s v%s at %s", model_name, version, weights_path)
        return entry

    def promote(self, model_name: str, version: str) -> Dict[str, Any]:
        models = self._registry.get("models", {})
        versions = models.get(model_name, [])

        target = None
        for v in versions:
            if v["version"] == version:
                target = v
                v["status"] = "active"
            else:
                if v.get("status") in ("active", "registered"):
                    v["status"] = "retired"

        if not target:
            raise ValueError(f"Version {version} not found for {model_name}")

        self._registry.setdefault("active", {})[model_name] = {
            "version": version,
            "weights_path": target["weights_path"],
        }

        self._save()
        logger.info("Promoted %s to v%s", model_name, version)
        return target

    def rollback(self, model_name: str) -> Optional[Dict[str, Any]]:
        models = self._registry.get("models", {})
        versions = models.get(model_name, [])

        active_idx = None
        for i, v in enumerate(versions):
            if v.get("status") == "active":
                active_idx = i
                break

        if active_idx is None or active_idx == 0:
            candidates = [v for v in versions if v.get("status") == "retired"]
            if not candidates:
                return None
            return self.promote(model_name, candidates[-1]["version"])

        previous = versions[active_idx - 1]
        return self.promote(model_name, previous["version"])

    def get_active(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self._registry.get("active", {}).get(model_name)

    def get_versions(self, model_name: str) -> List[Dict[str, Any]]:
        return self._registry.get("models", {}).get(model_name, [])

    def list_models(self) -> Dict[str, Any]:
        return {
            "models": list(self._registry.get("models", {}).keys()),
            "active": self._registry.get("active", {}),
        }
