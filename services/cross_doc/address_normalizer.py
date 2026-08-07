from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from services.extraction.normalize_extended import STATE_ALIASES, extract_pincode


_ABBREVIATIONS = {
    "rd": "road", "st": "street", "apt": "apartment", "blk": "block",
    "bldg": "building", "nr": "near", "opp": "opposite", "dist": "district",
    "tehsil": "tehsil", "po": "post office", "ps": "police station",
    "hno": "house number", "h.no": "house number", "vill": "village",
    "moh": "mohalla", "sec": "sector", "ph": "phase",
}


class IndianAddressNormalizer:
    """Normalize and compare Indian addresses."""

    def normalize(self, raw_address: str) -> Dict[str, Any]:
        if not raw_address:
            return {}

        s = re.sub(r"\s+", " ", raw_address).strip()

        pincode = extract_pincode(s)

        state = None
        for abbr, full_name in STATE_ALIASES.items():
            if full_name.lower() in s.lower() or f" {abbr} " in f" {s.upper()} ":
                state = full_name
                break

        city = None
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 3:
            city = parts[-2] if pincode and parts[-1].strip() == pincode else parts[-1]

        words = s.lower().split()
        expanded = []
        for w in words:
            clean = re.sub(r"[^a-z]", "", w)
            expanded.append(_ABBREVIATIONS.get(clean, w))
        normalized_text = " ".join(expanded)

        return {
            "raw": raw_address,
            "normalized": normalized_text,
            "pincode": pincode,
            "state": state,
            "city": city,
        }

    def compare(self, addr1: Dict[str, Any], addr2: Dict[str, Any]) -> float:
        if not addr1 or not addr2:
            return 0.0

        score = 0.0
        checks = 0

        pin1 = addr1.get("pincode")
        pin2 = addr2.get("pincode")
        if pin1 and pin2:
            checks += 1
            if pin1 == pin2:
                score += 1.0

        st1 = addr1.get("state")
        st2 = addr2.get("state")
        if st1 and st2:
            checks += 1
            if st1.lower() == st2.lower():
                score += 1.0

        n1 = addr1.get("normalized", "")
        n2 = addr2.get("normalized", "")
        if n1 and n2:
            checks += 1
            score += SequenceMatcher(None, n1, n2).ratio()

        return score / max(1, checks)
