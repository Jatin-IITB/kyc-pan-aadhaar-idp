from __future__ import annotations

import re
from copy import deepcopy
from difflib import get_close_matches
from typing import Any, Dict, Optional

FieldObj = Dict[str, Any]
Extraction = Dict[str, FieldObj]

_PAN_STRICT_REGEX = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_FULL_DATE_RE = re.compile(r"(?P<d>\d{1,2})[\/\-\.](?P<m>\d{1,2})[\/\-\.](?P<y>(?:19|20)\d{2})")
_YEAR_ONLY_RE = re.compile(r"\b((?:19|20)\d{2})\b")

_CONFUSABLE_DIGITS = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1", "|": "1",
    "Z": "2", "S": "5", "B": "8", "G": "6",
    "ION" : "10", "LZ" : "12", "VI" : "V1"
}
_CONFUSABLE_LETTERS = {
    "0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G",
}

_verhoeff_d = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0],
]
_verhoeff_p = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8],
]

def _verhoeff_check(num: str) -> bool:
    if not num.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _verhoeff_d[c][_verhoeff_p[i % 8][int(ch)]]
    return c == 0

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _set_value(field: FieldObj, value: str, by: Optional[str]) -> None:
    field["value"] = value
    if by:
        meta = field.setdefault("metadata", {})
        meta["source"] = by
        meta["is_normalized"] = True

def _fix_scientific(value: str) -> str:
    try:
        if "e" in value.lower():
            return str(int(float(value)))
    except Exception:
        pass
    return value

def normalize_gender(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^A-Za-z]", "", raw).upper()
    if s.startswith(("MALE", "MAN", "M")):
        return "Male"
    if s.startswith(("FEMALE", "WOMAN", "F")):
        return "Female"
    if s.startswith(("OTHER", "OTH", "TRANS", "T")):
        return "Other"
    m = get_close_matches(s, ["MALE", "FEMALE", "OTHER"], n=1, cutoff=0.6)
    return m[0].title() if m else None

def normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"(?i)^(date|dob|birth|of|year|yob|:|\s)+", "", raw.strip())
    m = _FULL_DATE_RE.search(s)
    if m:
        d, mth, y = int(m.group("d")), int(m.group("m")), m.group("y")
        if 1 <= d <= 31 and 1 <= mth <= 12:
            return f"{d:02d}/{mth:02d}/{y}"
    y = _YEAR_ONLY_RE.search(s)
    if y:
        return y.group(1)
    return None

def normalize_aadhaar_number(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^0-9A-Z]", "", _fix_scientific(raw).upper())
    if s.isdigit() and len(s) == 12 and _verhoeff_check(s):
        return s
    fixed = "".join(_CONFUSABLE_DIGITS.get(c, c) for c in s)
    digits = re.sub(r"\D", "", fixed)
    if len(digits) == 12 and _verhoeff_check(digits):
        return digits
    return None

def normalize_pan_number(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if len(s) != 10:
        return None
    chars = list(s)
    for i in range(10):
        if i < 5 or i == 9:
            if chars[i].isdigit():
                chars[i] = _CONFUSABLE_LETTERS.get(chars[i], chars[i])
        else:
            if not chars[i].isdigit():
                chars[i] = _CONFUSABLE_DIGITS.get(chars[i], chars[i])
    cand = "".join(chars)
    return cand if _PAN_STRICT_REGEX.fullmatch(cand) else None

def normalize_name(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^A-Za-z\s\.\'-]", "", raw)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

NORMALIZERS = {
    "gender": normalize_gender,
    "date_of_birth": normalize_date,
    "dob": normalize_date,
    "aadhaar_number": normalize_aadhaar_number,
    "pan_number": normalize_pan_number,
    "pan_num": normalize_pan_number,
    "name": normalize_name,
    "father_name": normalize_name,
}

def normalize_extraction(extraction: Extraction, doc_type: str) -> Extraction:
    out = deepcopy(extraction or {})
    for key, field in out.items():
        if not isinstance(field, dict):
            continue
        raw = _safe_str(field.get("value"))
        if not raw:
            continue
        fn = NORMALIZERS.get(key)
        if fn:
            nv = fn(raw)
            if nv:
                _set_value(field, nv, f"normalize_{key}")
    return out
