"""Synthetic Indian identity generation for the Truth Engine (Phase 11).

All identities are fabricated via Faker(en_IN). Document numbers are
structure-valid so they exercise the same validation paths as real documents:
PAN matches the official grammar, Aadhaar numbers carry a correct Verhoeff
check digit (the actual UIDAI algorithm).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict

import numpy as np
from faker import Faker

# --- Verhoeff checksum (UIDAI uses this for Aadhaar) ---------------------

_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_check_digit(number: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


def verhoeff_validate(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


# --- Identity + document-number generators -------------------------------

_HONORIFICS = ("Dr. ", "Mr. ", "Mrs. ", "Ms. ", "Miss ")
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_STATES = ["MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "WB", "MP", "HR"]
_BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]


def _clean_name(name: str) -> str:
    for h in _HONORIFICS:
        if name.startswith(h):
            name = name[len(h):]
    return name.upper().strip()


def make_person(rng: np.random.Generator, fake: Faker) -> Dict[str, Any]:
    gender = "Male" if rng.random() < 0.5 else "Female"
    raw = fake.name_male() if gender == "Male" else fake.name_female()
    name = _clean_name(raw)
    surname = name.split()[-1]
    father = _clean_name(fake.name_male())
    father_name = f"{father.split()[0]} {surname}"

    dob = fake.date_of_birth(minimum_age=18, maximum_age=70)
    address = fake.address().replace("\n", ", ")

    return {
        "name": name,
        "father_name": father_name,
        "gender": gender,
        "dob": dob.strftime("%d/%m/%Y"),
        "address": address,
        "surname": surname,
    }


def pan_number(rng: np.random.Generator, surname: str) -> str:
    # AAAPA1234A grammar: chars 1-3 random, char 4 entity code (P=person),
    # char 5 first letter of surname, then 4 digits + 1 letter.
    head = "".join(_LETTERS[rng.integers(0, 26)] for _ in range(3))
    digits = f"{rng.integers(0, 10000):04d}"
    tail = _LETTERS[rng.integers(0, 26)]
    return f"{head}P{surname[0]}{digits}{tail}"


def aadhaar_number(rng: np.random.Generator) -> str:
    # First digit 2-9 per UIDAI; 11 digits + Verhoeff check digit.
    body = str(rng.integers(2, 10)) + "".join(str(rng.integers(0, 10)) for _ in range(10))
    full = body + verhoeff_check_digit(body)
    return f"{full[0:4]} {full[4:8]} {full[8:12]}"


def dl_number(rng: np.random.Generator) -> str:
    state = _STATES[rng.integers(0, len(_STATES))]
    rto = rng.integers(1, 51)
    year = rng.integers(2005, 2024)
    serial = rng.integers(0, 10_000_000)
    return f"{state}{rto:02d} {year}{serial:07d}"


# --- Per-document field sets ---------------------------------------------

def pan_fields(person: Dict[str, Any], rng: np.random.Generator) -> Dict[str, str]:
    return {
        "pan_number": pan_number(rng, person["surname"]),
        "name": person["name"],
        "father_name": person["father_name"],
        "date_of_birth": person["dob"],
    }


def aadhaar_fields(person: Dict[str, Any], rng: np.random.Generator) -> Dict[str, str]:
    return {
        "aadhaar_number": aadhaar_number(rng),
        "name": person["name"],
        "date_of_birth": person["dob"],
        "gender": person["gender"],
        "address": person["address"],
    }


def dl_fields(person: Dict[str, Any], rng: np.random.Generator) -> Dict[str, str]:
    issue_year = int(rng.integers(2015, 2024))
    issue = dt.date(issue_year, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
    expiry = issue.replace(year=issue_year + 20)
    return {
        "dl_number": dl_number(rng),
        "name": person["name"],
        "date_of_birth": person["dob"],
        "blood_group": _BLOOD_GROUPS[rng.integers(0, len(_BLOOD_GROUPS))],
        "address": person["address"],
        "date_of_issue": issue.strftime("%d/%m/%Y"),
        "date_of_expiry": expiry.strftime("%d/%m/%Y"),
    }


FIELD_GENERATORS = {
    "pan": pan_fields,
    "aadhaar": aadhaar_fields,
    "driving_license": dl_fields,
}
