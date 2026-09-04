"""Metric primitives for the Truth Engine eval harness (Phase 11 W3)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from services.cross_doc.entity_resolver import _jaro_winkler

FUZZY_THRESHOLD = 0.90

# Fields where fuzzy matching is meaningless: one wrong digit in a PAN or DOB
# is a different identity, not a near-miss.
_EXACT_ONLY = re.compile(r"(number|date|dob|expiry|issue)", re.IGNORECASE)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().upper()


def field_match(field: str, truth: str, predicted: str) -> Tuple[bool, bool]:
    """Return (exact, fuzzy) match verdicts for one field."""
    t, p = normalize(truth), normalize(predicted)
    exact = bool(t) and t == p
    if exact:
        return True, True
    if _EXACT_ONLY.search(field):
        return exact, exact
    fuzzy = bool(t) and bool(p) and _jaro_winkler(t, p) >= FUZZY_THRESHOLD
    return exact, fuzzy


def prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def score_extraction(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score (truth, predicted) field dicts across samples.

    IE convention per field: non-empty & correct = TP, non-empty & wrong = FP,
    empty prediction with non-empty truth = FN.
    """
    per_field: Dict[str, Dict[str, int]] = {}
    totals = {"exact": [0, 0, 0], "fuzzy": [0, 0, 0]}  # tp, fp, fn

    for s in samples:
        truth, pred = s["truth"], s["predicted"]
        for field, want in truth.items():
            got = str(pred.get(field, "") or "")
            exact, fuzzy = field_match(field, want, got)
            slot = per_field.setdefault(field, {"tp": 0, "fp": 0, "fn": 0, "fuzzy_tp": 0})
            if not normalize(got):
                slot["fn"] += 1
                totals["exact"][2] += 1
                totals["fuzzy"][2] += 1
                continue
            slot["tp" if exact else "fp"] += 1
            totals["exact"][0 if exact else 1] += 1
            slot["fuzzy_tp"] += int(fuzzy)
            totals["fuzzy"][0 if fuzzy else 1] += 1

    per_field_scores = {
        f: {**prf1(c["tp"], c["fp"], c["fn"]),
            "fuzzy_f1": prf1(c["fuzzy_tp"], c["tp"] + c["fp"] - c["fuzzy_tp"], c["fn"])["f1"],
            "n": c["tp"] + c["fp"] + c["fn"]}
        for f, c in sorted(per_field.items())
    }
    return {
        "n_samples": len(samples),
        "micro": prf1(*totals["exact"]),
        "micro_fuzzy": prf1(*totals["fuzzy"]),
        "per_field": per_field_scores,
    }


def check_gates(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Compare metrics against eval_thresholds.yaml. Gates for tiers that did
    not run are reported as skipped, not failed."""
    results: List[Dict[str, Any]] = []

    def gate(name: str, actual: Optional[float], limit: float, direction: str) -> None:
        if actual is None:
            results.append({"gate": name, "limit": limit, "actual": None, "status": "SKIPPED"})
            return
        ok = actual <= limit if direction == "max" else actual >= limit
        results.append({"gate": name, "limit": limit, "actual": round(float(actual), 4),
                        "status": "PASS" if ok else "FAIL"})

    forensics = metrics.get("forensics") or {}
    fth = thresholds.get("forensics", {})
    if "genuine_fpr_max" in fth:
        gate("forensics.genuine_fpr", forensics.get("genuine_fpr"), fth["genuine_fpr_max"], "max")
    if "overall_recall_min" in fth:
        gate("forensics.overall_recall", forensics.get("overall_recall"), fth["overall_recall_min"], "min")
    for attack, floor in (fth.get("per_attack_recall_min") or {}).items():
        actual = (forensics.get("per_attack") or {}).get(attack, {}).get("recall")
        gate(f"forensics.recall.{attack}", actual, floor, "min")

    decision = metrics.get("decision") or {}
    dth = thresholds.get("decision", {})
    if "flagged_leakage_max" in dth:
        gate("decision.flagged_leakage", decision.get("flagged_leakage"),
             dth["flagged_leakage_max"], "max")
    if "undetected_autoclear_max" in dth:
        gate("decision.undetected_autoclear", decision.get("undetected_autoclear"),
             dth["undetected_autoclear_max"], "max")
    if "genuine_auto_clear_min" in dth:
        gate("decision.genuine_auto_clear", decision.get("genuine_auto_clear_rate"),
             dth["genuine_auto_clear_min"], "min")

    extraction = metrics.get("extraction")
    eth = thresholds.get("extraction", {})
    if "micro_f1_min" in eth:
        gate("extraction.micro_f1",
             (extraction or {}).get("micro", {}).get("f1") if extraction else None,
             eth["micro_f1_min"], "min")
    if "fuzzy_f1_min" in eth:
        gate("extraction.fuzzy_f1",
             (extraction or {}).get("micro_fuzzy", {}).get("f1") if extraction else None,
             eth["fuzzy_f1_min"], "min")

    rag = metrics.get("rag")
    rth = thresholds.get("rag", {})
    for metric_name in (
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "negative_abstention_rate",
        "citation_support_rate",
        "judge_agreement",
    ):
        threshold_name = f"{metric_name}_min"
        if threshold_name in rth:
            gate(
                f"rag.{metric_name}",
                (rag or {}).get(metric_name) if rag else None,
                rth[threshold_name],
                "min",
            )

    passed = all(r["status"] != "FAIL" for r in results)
    return {"passed": passed, "results": results}
