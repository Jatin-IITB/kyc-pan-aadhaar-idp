from __future__ import annotations

from typing import Any, Dict, List

from services.cross_doc.entity_resolver import EntityResolver


class ContradictionDetector:
    """Find contradictions across documents in a KYC packet."""

    def __init__(self) -> None:
        self.resolver = EntityResolver()

    def detect(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(documents) < 2:
            return {"contradictions": [], "consistency_score": 1.0, "recommendation": "PASS"}

        contradictions: List[Dict[str, Any]] = []

        names = []
        dobs = []
        genders = []

        for doc in documents:
            doc_type = doc.get("doc_type", "unknown")
            fields = doc.get("fields", {})

            for name_key in ("name", "surname", "given_names", "customer_name", "account_holder", "employee_name"):
                if fields.get(name_key):
                    names.append({"doc_type": doc_type, "value": str(fields[name_key])})

            if fields.get("date_of_birth"):
                dobs.append({"doc_type": doc_type, "value": str(fields["date_of_birth"])})

            if fields.get("gender"):
                genders.append({"doc_type": doc_type, "value": str(fields["gender"])})

        if len(names) >= 2:
            name_result = self.resolver.resolve_names(names)
            for m in name_result.get("mismatches", []):
                contradictions.append({
                    "type": "name_mismatch",
                    "severity": "HIGH" if m.get("score", 0) < 0.70 else "MEDIUM",
                    "detail": f"{m['values'][0]} vs {m['values'][1]}",
                    "docs": m.get("docs", []),
                })

        if len(dobs) >= 2:
            dob_result = self.resolver.resolve_dates(dobs)
            for m in dob_result.get("mismatches", []):
                contradictions.append({
                    "type": "dob_mismatch",
                    "severity": "CRITICAL",
                    "detail": f"{m['values'][0]} vs {m['values'][1]}",
                    "docs": m.get("docs", []),
                })

        if len(genders) >= 2:
            unique_genders = set(g["value"].upper() for g in genders)
            if len(unique_genders) > 1:
                contradictions.append({
                    "type": "gender_mismatch",
                    "severity": "HIGH",
                    "detail": f"Conflicting genders: {', '.join(unique_genders)}",
                    "docs": [g["doc_type"] for g in genders],
                })

        n_checks = max(1, len(names) + len(dobs) + len(genders))
        n_issues = len(contradictions)
        consistency_score = max(0.0, 1.0 - (n_issues / n_checks))

        if any(c["severity"] == "CRITICAL" for c in contradictions):
            recommendation = "REJECT"
        elif contradictions:
            recommendation = "REVIEW"
        else:
            recommendation = "PASS"

        return {
            "contradictions": contradictions,
            "consistency_score": consistency_score,
            "recommendation": recommendation,
        }
