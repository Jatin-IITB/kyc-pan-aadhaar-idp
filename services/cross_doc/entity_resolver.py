from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


def _jaro_winkler(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    max_dist = max(len(s1), len(s2)) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    transpositions = 0

    for i in range(len(s1)):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len(s2))
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len(s1) + matches / len(s2) + (matches - transpositions / 2) / matches) / 3

    prefix_len = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * 0.1 * (1 - jaro)


def _soundex(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"[^A-Z]", "", name.upper())
    if not name:
        return ""

    codes = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    result = name[0]
    prev_code = codes.get(name[0], "0")

    for ch in name[1:]:
        code = codes.get(ch, "0")
        if code != "0" and code != prev_code:
            result += code
        prev_code = code if code != "0" else prev_code

    return (result + "000")[:4]


def _token_set_ratio(s1: str, s2: str) -> float:
    tokens1 = set(s1.upper().split())
    tokens2 = set(s2.upper().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


class EntityResolver:
    """Fuzzy entity matching across documents in a KYC packet."""

    def resolve_names(self, names: List[Dict[str, str]]) -> Dict[str, Any]:
        if len(names) < 2:
            return {"match_score": 1.0, "canonical_name": names[0]["value"] if names else "", "mismatches": [], "is_same_person": True}

        scores = []
        mismatches = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1 = names[i]["value"].upper().strip()
                n2 = names[j]["value"].upper().strip()

                jw = _jaro_winkler(n1, n2)
                sx_match = _soundex(n1) == _soundex(n2)
                tsr = _token_set_ratio(n1, n2)
                seq = SequenceMatcher(None, n1, n2).ratio()

                combined = 0.3 * jw + 0.1 * (1.0 if sx_match else 0.0) + 0.3 * tsr + 0.3 * seq
                scores.append(combined)

                if combined < 0.85:
                    detail = "phonetic variant" if sx_match else "name mismatch"
                    mismatches.append({
                        "field": "name",
                        "docs": [names[i].get("doc_type", ""), names[j].get("doc_type", "")],
                        "values": [n1, n2],
                        "detail": detail,
                        "score": combined,
                    })

        avg_score = sum(scores) / len(scores) if scores else 1.0
        canonical = max(names, key=lambda n: len(n["value"]))["value"]

        return {
            "match_score": avg_score,
            "canonical_name": canonical,
            "mismatches": mismatches,
            "is_same_person": avg_score >= 0.65,
        }

    _DATE_FORMATS = (
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%d %b %Y", "%d %B %Y",
        "%b %d, %Y", "%B %d, %Y",
    )

    @classmethod
    def _parse_date(cls, raw: str) -> Optional[datetime]:
        cleaned = raw.strip()
        for fmt in cls._DATE_FORMATS:
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None

    def resolve_dates(self, dates: List[Dict[str, str]]) -> Dict[str, Any]:
        if len(dates) < 2:
            return {"match": True, "mismatches": []}

        mismatches = []
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                d1 = dates[i]["value"].strip()
                d2 = dates[j]["value"].strip()

                parsed_d1 = self._parse_date(d1)
                parsed_d2 = self._parse_date(d2)

                if parsed_d1 and parsed_d2:
                    match = parsed_d1.date() == parsed_d2.date()
                else:
                    match = d1 == d2

                if not match:
                    mismatches.append({
                        "field": "date_of_birth",
                        "docs": [dates[i].get("doc_type", ""), dates[j].get("doc_type", "")],
                        "values": [d1, d2],
                        "severity": "CRITICAL",
                    })

        return {"match": len(mismatches) == 0, "mismatches": mismatches}
