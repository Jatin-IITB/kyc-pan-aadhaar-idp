from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RegressionChecker:
    """Compare model performance against a held-out eval set to detect regressions."""

    def __init__(self, eval_set_path: str = "data/eval_set") -> None:
        self.eval_root = Path(eval_set_path)

    def load_eval_set(self) -> List[Dict[str, Any]]:
        if not self.eval_root.exists():
            return []

        samples = []
        for f in sorted(self.eval_root.glob("*.json")):
            samples.append(json.loads(f.read_text()))
        return samples

    def evaluate_model(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not predictions or not ground_truth:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "field_scores": {}}

        gt_by_id = {g.get("document_id", str(i)): g for i, g in enumerate(ground_truth)}

        tp = 0
        fp = 0
        fn = 0
        field_tp: Dict[str, int] = {}
        field_fp: Dict[str, int] = {}
        field_fn: Dict[str, int] = {}

        for pred in predictions:
            doc_id = pred.get("document_id", "")
            gt = gt_by_id.get(doc_id)
            if not gt:
                continue

            pred_fields = pred.get("fields", {})
            gt_fields = gt.get("fields", {})

            all_keys = set(pred_fields.keys()) | set(gt_fields.keys())
            for key in all_keys:
                p_val = str(pred_fields.get(key, "")).strip()
                g_val = str(gt_fields.get(key, "")).strip()

                if p_val and g_val and p_val == g_val:
                    tp += 1
                    field_tp[key] = field_tp.get(key, 0) + 1
                elif p_val and (not g_val or p_val != g_val):
                    fp += 1
                    field_fp[key] = field_fp.get(key, 0) + 1
                elif g_val and not p_val:
                    fn += 1
                    field_fn[key] = field_fn.get(key, 0) + 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        field_scores = {}
        for key in set(list(field_tp.keys()) + list(field_fp.keys()) + list(field_fn.keys())):
            ftp = field_tp.get(key, 0)
            ffp = field_fp.get(key, 0)
            ffn = field_fn.get(key, 0)
            fp_ = ftp / (ftp + ffp) if (ftp + ffp) > 0 else 0.0
            fr_ = ftp / (ftp + ffn) if (ftp + ffn) > 0 else 0.0
            ff1 = 2 * fp_ * fr_ / (fp_ + fr_) if (fp_ + fr_) > 0 else 0.0
            field_scores[key] = {"precision": fp_, "recall": fr_, "f1": ff1}

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "field_scores": field_scores,
            "counts": {"tp": tp, "fp": fp, "fn": fn},
        }

    def check_regression(
        self,
        new_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        f1_threshold: float = 0.02,
    ) -> Dict[str, Any]:
        new_f1 = new_metrics.get("f1", 0.0)
        baseline_f1 = baseline_metrics.get("f1", 0.0)
        drop = baseline_f1 - new_f1

        regressed_fields = []
        new_fields = new_metrics.get("field_scores", {})
        base_fields = baseline_metrics.get("field_scores", {})
        for key in base_fields:
            if key in new_fields:
                field_drop = base_fields[key].get("f1", 0.0) - new_fields[key].get("f1", 0.0)
                if field_drop > f1_threshold:
                    regressed_fields.append({
                        "field": key,
                        "baseline_f1": base_fields[key]["f1"],
                        "new_f1": new_fields[key]["f1"],
                        "drop": field_drop,
                    })

        is_regression = drop > f1_threshold or len(regressed_fields) > 0

        return {
            "is_regression": is_regression,
            "overall_f1_drop": drop,
            "baseline_f1": baseline_f1,
            "new_f1": new_f1,
            "regressed_fields": regressed_fields,
            "recommendation": "ROLLBACK" if is_regression else "PROMOTE",
        }
