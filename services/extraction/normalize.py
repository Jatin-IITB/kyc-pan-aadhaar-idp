# services/extraction/normalize.py
from __future__ import annotations

import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional


FieldObj = Dict[str, Any]
Extraction = Dict[str, FieldObj]


_DIGITS_RE = re.compile(r"\d+")
_NON_LETTERS_RE = re.compile(r"[^A-Za-z]+")


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _set_value(field: FieldObj, new_value: str) -> None:
    # Keep det_conf/ocr_conf/bbox untouched; just update the value.
    field["value"] = new_value


def normalize_extraction(extraction: Extraction, doc_type: str) -> Extraction:
    """
    Normalize OCR outputs into canonical formats *before* schema validation.

    Expected extraction shape:
      extraction[field] = { "value": str, "det_conf": float, "ocr_conf": float, "bbox": [...] }

    Returns a deep-copied, normalized extraction dict.
    """
    out: Extraction = deepcopy(extraction or {})
    dt = _safe_str(doc_type).lower()

    # Common (doc-agnostic) normalizations can go here if needed.

    if dt == "aadhaar":
        _normalize_aadhaar(out)
    elif dt == "pan":
        _normalize_pan(out)

    return out


def _normalize_aadhaar(ex: Extraction) -> None:
    # aadhaar_number: keep exactly 12 digits if possible.
    if "aadhaar_number" in ex and isinstance(ex["aadhaar_number"], dict):
        v = _safe_str(ex["aadhaar_number"].get("value"))
        nv = normalize_aadhaar_number(v)
        if nv is not None:
            _set_value(ex["aadhaar_number"], nv)

    # date_of_birth: DD/MM/YYYY (based on your schema validation errors).
    if "date_of_birth" in ex and isinstance(ex["date_of_birth"], dict):
        v = _safe_str(ex["date_of_birth"].get("value"))
        nv = normalize_dob_aadhaar(v)
        if nv is not None:
            _set_value(ex["date_of_birth"], nv)

    # gender: map OCR variants -> allowed enum tokens.
    if "gender" in ex and isinstance(ex["gender"], dict):
        v = _safe_str(ex["gender"].get("value"))
        nv = normalize_gender(v)
        if nv is not None:
            _set_value(ex["gender"], nv)


def _normalize_pan(ex: Extraction) -> None:
    # PAN DOB in your runs is consistently accepted as digits like DDMMYYYY.
    if "date_of_birth" in ex and isinstance(ex["date_of_birth"], dict):
        v = _safe_str(ex["date_of_birth"].get("value"))
        nv = normalize_dob_pan(v)
        if nv is not None:
            _set_value(ex["date_of_birth"], nv)

    # PAN number itself usually OK; optionally could uppercase+strip spaces.
    if "pan_number" in ex and isinstance(ex["pan_number"], dict):
        v = _safe_str(ex["pan_number"].get("value"))
        if v:
            _set_value(ex["pan_number"], v.replace(" ", "").upper())


def normalize_aadhaar_number(v: str, pad_left_if_short: bool = True) -> Optional[str]:
    """
    Attempts to convert:
      - 'XXXX XXXX XXXX' -> 'XXXXXXXXXXXX'
      - '9.387e+10' (scientific notation) -> digits (optionally left-pad to 12)

    Returns None if no safe normalization found.
    """
    s = _safe_str(v)
    if not s:
        return None

    # First, common case: digits with spaces.
    digits = "".join(_DIGITS_RE.findall(s))
    if len(digits) == 12:
        return digits

    # If it looks like scientific notation or a float, try Decimal parsing.
    if any(ch in s.lower() for ch in ("e", ".")):
        try:
            d = Decimal(s)
        except InvalidOperation:
            return digits if len(digits) == 12 else None

        # Must be an integer-like number.
        if d != d.to_integral_value():
            return digits if len(digits) == 12 else None

        as_int = str(int(d))
        if len(as_int) == 12:
            return as_int
        if pad_left_if_short and len(as_int) < 12:
            return as_int.zfill(12)
        return None

    # If it’s 11 digits, optionally pad (helps when leading zero got dropped).
    if pad_left_if_short and len(digits) == 11:
        return digits.zfill(12)

    return None


def normalize_gender(v: str) -> Optional[str]:
    """
    Normalizes common OCR variants to one of: 'MALE', 'FEMALE', 'Other'.
    (Your schema accepts case variants, so returning uppercase is safe.)
    """
    s = _safe_str(v)
    if not s:
        return None

    # Keep only letters to handle 'I MALE', 'Femaie', etc.
    letters = _NON_LETTERS_RE.sub("", s).upper()

    # Common OCR confusions.
    if letters in ("IMALE", "LMALE", "1MALE"):
        letters = "MALE"
    if letters in ("FEMAIE", "FEMALF", "FEMALC"):
        letters = "FEMALE"

    if "FEMALE" in letters or letters.startswith("FEM"):
        return "FEMALE"
    if "MALE" in letters or letters.endswith("MALE") or letters.startswith("MAL"):
        return "MALE"
    if letters in ("OTHER", "OTH"):
        return "Other"

    return None


def normalize_dob_aadhaar(v: str) -> Optional[str]:
    """
    Aadhaar DOB canonical: DD/MM/YYYY.

    Handles:
      - 'DDMMYYYY' -> 'DD/MM/YYYY'
      - 'DD-MM-YYYY' -> 'DD/MM/YYYY'
      - 'DD/MMYYYY' -> 'DD/MM/YYYY'  (e.g., 15/082001)
      - 'DDMM/YYYY' -> 'DD/MM/YYYY'  (e.g., 0108/1973)
    """
    s = _safe_str(v)
    if not s:
        return None

    # Already correct-ish: DD/MM/YYYY
    if re.fullmatch(r"(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}", s):
        return s

    # DD-MM-YYYY or DD.MM.YYYY -> DD/MM/YYYY
    m = re.fullmatch(r"(\d{2})[-.](\d{2})[-.](\d{4})", s)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{dd}/{mm}/{yyyy}"

    # DDMMYYYY -> DD/MM/YYYY
    digits = "".join(_DIGITS_RE.findall(s))
    if len(digits) == 8:
        dd, mm, yyyy = digits[:2], digits[2:4], digits[4:]
        return f"{dd}/{mm}/{yyyy}"

    # DD/MMYYYY (2 + 1 + 6) -> DD/MM/YYYY
    m = re.fullmatch(r"(\d{2})/(\d{6})", s)
    if m:
        dd = m.group(1)
        mm = m.group(2)[:2]
        yyyy = m.group(2)[2:]
        return f"{dd}/{mm}/{yyyy}"

    # DDMM/YYYY (4 + 1 + 4) -> DD/MM/YYYY
    m = re.fullmatch(r"(\d{4})/(\d{4})", s)
    if m:
        dd = m.group(1)[:2]
        mm = m.group(1)[2:]
        yyyy = m.group(2)
        return f"{dd}/{mm}/{yyyy}"

    # Not enough information (e.g., year-only) -> leave as-is.
    return None


def normalize_dob_pan(v: str) -> Optional[str]:
    """
    PAN DOB canonical: DD/MM/YYYY (to match pan.schema.json).

    Handles:
      - 'DDMMYYYY' -> 'DD/MM/YYYY'
      - 'DD-MM-YYYY' -> 'DD/MM/YYYY'
      - 'DD/MM/YYYY' -> unchanged
      - 'DDMMYY' -> DD/MM/YYYY with century heuristic
    """
    s = _safe_str(v)
    if not s:
        return None

    # Already correct: DD/MM/YYYY
    if re.fullmatch(r"(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}", s):
        return s

    digits = "".join(_DIGITS_RE.findall(s))

    # DDMMYYYY -> DD/MM/YYYY
    if len(digits) == 8:
        dd, mm, yyyy = digits[:2], digits[2:4], digits[4:]
        return f"{dd}/{mm}/{yyyy}"

    # DD-MM-YYYY / DD/MM/YYYY (robust)
    m = re.fullmatch(r"(\d{2})[-/](\d{2})[-/](\d{4})", s)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{dd}/{mm}/{yyyy}"

    # DDMMYY -> DD/MM/YYYY
    if len(digits) == 6:
        dd, mm, yy = digits[:2], digits[2:4], digits[4:]
        yy_i = int(yy)
        yyyy = f"19{yy}" if yy_i >= 30 else f"20{yy}"
        return f"{dd}/{mm}/{yyyy}"

    return None

