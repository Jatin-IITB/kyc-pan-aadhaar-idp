from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GroundTruthDB:
    """Persistent store for human-reviewed corrections used as training data."""

    def __init__(self, db_path: str = "data/ground_truth") -> None:
        self.root = Path(db_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._corrections_file = self.root / "corrections.jsonl"
        self._stats_file = self.root / "stats.json"

    def ingest_correction(
        self,
        document_id: str,
        doc_type: str,
        field_name: str,
        original_value: str,
        corrected_value: str,
        reviewer: str,
        image_uri: Optional[str] = None,
        extraction_path: str = "yolo",
    ) -> Dict[str, Any]:
        record = {
            "document_id": document_id,
            "doc_type": doc_type,
            "field_name": field_name,
            "original_value": original_value,
            "corrected_value": corrected_value,
            "reviewer": reviewer,
            "image_uri": image_uri,
            "extraction_path": extraction_path,
        }

        with open(self._corrections_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        self._update_stats(doc_type, field_name)
        return record

    def _update_stats(self, doc_type: str, field_name: str) -> None:
        stats = self._load_stats()
        stats["total_corrections"] = stats.get("total_corrections", 0) + 1

        by_type = stats.setdefault("by_doc_type", {})
        by_type[doc_type] = by_type.get(doc_type, 0) + 1

        by_field = stats.setdefault("by_field", {})
        by_field[field_name] = by_field.get(field_name, 0) + 1

        with open(self._stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    def _load_stats(self) -> Dict[str, Any]:
        if self._stats_file.exists():
            return json.loads(self._stats_file.read_text())
        return {"total_corrections": 0, "by_doc_type": {}, "by_field": {}}

    def get_stats(self) -> Dict[str, Any]:
        return self._load_stats()

    def get_corrections(self, doc_type: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        if not self._corrections_file.exists():
            return []

        records = []
        with open(self._corrections_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if doc_type and record.get("doc_type") != doc_type:
                    continue
                records.append(record)
                if len(records) >= limit:
                    break
        return records

    def export_training_set(self, output_path: str, doc_type: Optional[str] = None) -> Dict[str, Any]:
        corrections = self.get_corrections(doc_type=doc_type, limit=100_000)
        if not corrections:
            return {"exported": 0, "path": output_path}

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            for record in corrections:
                f.write(json.dumps(record, default=str) + "\n")

        return {"exported": len(corrections), "path": output_path}

    def error_distribution(self) -> Dict[str, Any]:
        corrections = self.get_corrections(limit=100_000)
        if not corrections:
            return {"total": 0, "by_field": {}, "by_doc_type": {}, "top_errors": []}

        field_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        error_pairs: Dict[str, int] = {}

        for c in corrections:
            field = c.get("field_name", "unknown")
            dt = c.get("doc_type", "unknown")
            field_counts[field] = field_counts.get(field, 0) + 1
            type_counts[dt] = type_counts.get(dt, 0) + 1

            pair_key = f"{dt}:{field}"
            error_pairs[pair_key] = error_pairs.get(pair_key, 0) + 1

        top_errors = sorted(error_pairs.items(), key=lambda x: -x[1])[:20]

        return {
            "total": len(corrections),
            "by_field": field_counts,
            "by_doc_type": type_counts,
            "top_errors": [{"key": k, "count": v} for k, v in top_errors],
        }
