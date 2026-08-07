from __future__ import annotations

import re
from typing import Optional


_PASSPORT_RE = re.compile(r"^[A-Z]\d{7}$")
_DL_RE = re.compile(r"^[A-Z]{2}\d{13}$")
_EPIC_RE = re.compile(r"^[A-Z]{3}\d{7}$")
_IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

STATE_ALIASES = {
    "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar", "CG": "Chhattisgarh", "GA": "Goa",
    "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh",
    "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala",
    "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
    "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland",
    "OD": "Odisha", "OR": "Odisha", "PB": "Punjab",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TS": "Telangana", "TR": "Tripura", "UP": "Uttar Pradesh",
    "UK": "Uttarakhand", "WB": "West Bengal",
    "DL": "Delhi", "JK": "Jammu and Kashmir", "LA": "Ladakh",
}

_PINCODE_RE = re.compile(r"\b(\d{6})\b")


def normalize_passport_number(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if len(s) == 8 and _PASSPORT_RE.fullmatch(s):
        return s
    return None


def normalize_dl_number(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[\s\-/]", "", raw.upper())
    s = re.sub(r"[^A-Z0-9]", "", s)
    if len(s) == 15 and _DL_RE.fullmatch(s):
        return s
    return s if len(s) >= 10 else None


def normalize_voter_id(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if len(s) == 10 and _EPIC_RE.fullmatch(s):
        return s
    return s if len(s) >= 8 else None


def normalize_ifsc(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if len(s) == 11 and _IFSC_RE.fullmatch(s):
        return s
    return None


def normalize_address_india(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"\s+", " ", raw).strip()
    s = re.sub(r",\s*,", ",", s)
    return s if len(s) >= 10 else None


def extract_pincode(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = _PINCODE_RE.search(raw)
    if m:
        pin = m.group(1)
        if pin[0] in "123456789":
            return pin
    return None


def normalize_amount(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^\d.,]", "", raw)
    return s if s else None
