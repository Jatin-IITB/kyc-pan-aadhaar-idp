from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Optional

FieldObj = Dict[str, Any]
Extraction = Dict[str, FieldObj]

_DIGITS_RE = re.compile(r"\d+")

# Strict validation
_PAN_STRICT_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# Confusable character maps (safe subset only)
_CONFUSABLE_DIGITS = {
    "O": "0", "o": "0", "D": "0",
    "I": "1", "l": "1", "L": "1", "|": "1",
    "Z": "2", "S": "5", "B": "8", "G": "6"
}
_CONFUSABLE_LETTERS = {
    "0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"
}

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _set_value(field: FieldObj, new_value: str) -> None:
    field["value"] = new_value

def normalize_extraction(extraction: Extraction, doc_type: str) -> Extraction:
    out: Extraction = deepcopy(extraction or {})
    dt = _safe_str(doc_type).lower()

    if dt == "aadhaar":
        _normalize_aadhaar(out)
    elif dt == "pan":
        _normalize_pan(out)

    return out

def _clean_pan_str(s: str) -> str:
    """
    Positional correction for PAN: ABCDE1234F
    Corrects only confusable characters by position.
    """
    s = s.replace(" ", "").replace("-", "").upper()
    if len(s) != 10:
        return s

    res = []

    # First 5: letters
    for i in range(5):
        ch = s[i]
        res.append(_CONFUSABLE_LETTERS.get(ch, ch) if ch.isdigit() else ch)

    # Next 4: digits
    for i in range(5, 9):
        ch = s[i]
        res.append(_CONFUSABLE_DIGITS.get(ch, ch) if not ch.isdigit() else ch)

    # Last: letter
    ch = s[9]
    res.append(_CONFUSABLE_LETTERS.get(ch, ch) if ch.isdigit() else ch)

    return "".join(res)

def _normalize_aadhaar(ex: Extraction) -> None:
    # Aadhaar number
    if "aadhaar_number" in ex and isinstance(ex["aadhaar_number"], dict):
        v = _safe_str(ex["aadhaar_number"].get("value"))
        nv = normalize_aadhaar_number(v)
        if nv is not None:
            _set_value(ex["aadhaar_number"], nv)

    # DOB (safe: no hallucination)
    if "date_of_birth" in ex and isinstance(ex["date_of_birth"], dict):
        v = _safe_str(ex["date_of_birth"].get("value"))
        nv = normalize_dob_aadhaar(v)
        if nv is not None:
            _set_value(ex["date_of_birth"], nv)

    # Gender (case normalization only)
    if "gender" in ex and isinstance(ex["gender"], dict):
        v = _safe_str(ex["gender"].get("value"))
        nv = normalize_gender(v)
        if nv is not None:
            _set_value(ex["gender"], nv)

def _normalize_pan(ex: Extraction) -> None:
    # DOB
    if "date_of_birth" in ex and isinstance(ex["date_of_birth"], dict):
        v = _safe_str(ex["date_of_birth"].get("value"))
        nv = normalize_dob_pan(v)
        if nv is not None:
            _set_value(ex["date_of_birth"], nv)

    # PAN number
    if "pan_number" in ex and isinstance(ex["pan_number"], dict):
        v = _safe_str(ex["pan_number"].get("value"))
        nv = _clean_pan_str(v)
        if nv and _PAN_STRICT_REGEX.fullmatch(nv):
            _set_value(ex["pan_number"], nv)

def normalize_aadhaar_number(v: str) -> Optional[str]:
    digits = "".join(_DIGITS_RE.findall(_safe_str(v)))
    return digits if len(digits) == 12 else None

def normalize_gender(v: str) -> Optional[str]:
    s = _safe_str(v).upper()
    if "FEMALE" in s or s.startswith("FEM"):
        return "Female"
    if "MALE" in s:
        return "Male"
    if "OTHER" in s:
        return "Other"
    return None

def normalize_dob_aadhaar(v: str) -> Optional[str]:
    """
    - Normalizes valid dates to DD/MM/YYYY
    - Allows YYYY to pass through unchanged
    - Never invents day/month
    """
    s = _safe_str(v)

    if re.fullmatch(r"(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}", s):
        return s

    digits = "".join(_DIGITS_RE.findall(s))

    if len(digits) == 8:
        return f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"

    if len(digits) == 6:
        dd, mm, yy = digits[:2], digits[2:4], digits[4:]
        yyyy = f"19{yy}" if int(yy) > 30 else f"20{yy}"
        return f"{dd}/{mm}/{yyyy}"

    if re.fullmatch(r"\d{4}", s):
        return s

    return None

def normalize_dob_pan(v: str) -> Optional[str]:
    s = _safe_str(v)

    if re.fullmatch(r"(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}", s):
        return s

    digits = "".join(_DIGITS_RE.findall(s))
    if len(digits) == 8:
        return f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"

    return None
