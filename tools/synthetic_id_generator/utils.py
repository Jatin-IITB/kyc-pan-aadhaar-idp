import random
import string
from datetime import datetime, timedelta
from typing import Tuple, List
import numpy as np
import cv2
from faker import Faker

fake = Faker("en_IN")


def generate_pan_number(status_code: str = "P", name: str | None = None) -> str:
    """
    Generate valid PAN number: [AAA][P/C/H/etc][A][1234][A]
    The 5th character (for 'P') is the first letter of surname (CRITICAL for realism).
    """
    first_three = "".join(random.choices(string.ascii_uppercase, k=3))
    fourth = status_code
    
    # 5th char: surname initial (if name provided and status is Individual)
    fifth = random.choice(string.ascii_uppercase)
    if status_code == "P" and name:
        surname_parts = name.split()
        if surname_parts:
            surname = surname_parts[-1]
            if surname and surname.isalpha():
                fifth = surname[0].upper()

    numbers = "".join(str(random.randint(0, 9)) for _ in range(4))
    tenth = random.choice(string.ascii_uppercase)
    
    return f"{first_three}{fourth}{fifth}{numbers}{tenth}"


def compute_verhoeff_checksum(digits: List[int]) -> int:
    """Verhoeff checksum (official Aadhaar algorithm)."""
    # CORRECT d_table (multiplication table)
    d_table = [
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
    
    # CORRECT p_table (permutation table)
    p_table = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
    ]
    
    inv_table = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

    c = 0
    for i, digit in enumerate(reversed(digits)):
        c = d_table[c][p_table[(i + 1) % 8][digit]]
    
    return inv_table[c]


def verify_verhoeff_checksum(aadhaar_number: str) -> bool:
    """Verify 12-digit Aadhaar using Verhoeff algorithm."""
    if len(aadhaar_number) != 12 or not aadhaar_number.isdigit():
        return False
    
    digits = [int(d) for d in aadhaar_number]
    
    # For validation: checksum of all 12 digits should be 0
    d_table = [
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
    
    p_table = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
    ]

    c = 0
    for i, digit in enumerate(reversed(digits)):
        c = d_table[c][p_table[i % 8][digit]]

    return c == 0


def generate_aadhaar_number() -> str:
    """Generate valid 12-digit Aadhaar with Verhoeff checksum."""
    # First digit: 2-9 (Aadhaar spec)
    first_digit = random.randint(2, 9)
    remaining = [random.randint(0, 9) for _ in range(10)]
    
    base_digits = [first_digit] + remaining
    checksum = compute_verhoeff_checksum(base_digits)
    
    return "".join(map(str, base_digits)) + str(checksum)


def generate_indian_name(gender: str | None = None) -> Tuple[str, str]:
    """Generate realistic Indian name (uppercase for ID realism)."""
    if gender is None:
        gender = random.choice(["Male", "Female"])
    
    if gender == "Male":
        first = fake.first_name_male()
        father_first = fake.first_name_male()
    else:
        first = fake.first_name_female()
        father_first = fake.first_name_male()
    
    last = fake.last_name()
    
    name = f"{first} {last}".upper()
    father_name = f"{father_first} {last}".upper()
    
    return name, father_name


def generate_indian_address() -> str:
    """Generate realistic Indian address."""
    components = [
        f"H.No. {random.randint(10, 999)}",
        fake.street_name(),
        fake.city(),
        fake.state(),
        f"PIN {fake.postcode()}",
    ]
    return ", ".join(components)


def generate_dob(min_age: int = 18, max_age: int = 80) -> str:
    """Generate DD/MM/YYYY formatted DOB."""
    today = datetime.now()
    age_days = random.randint(min_age * 365, max_age * 365)
    dob = today - timedelta(days=age_days)
    return dob.strftime("%d/%m/%Y")


def choose_style(weights: Tuple[float, float]) -> str:
    """Choose 'scan' or 'phone' based on weights."""
    return random.choices(["scan", "phone"], weights=weights, k=1)[0]


def generate_face_photo(size_wh: Tuple[int, int], seed: int | None = None) -> np.ndarray:
    """
    Generate synthetic ID photo placeholder (non-PII).
    Returns BGR image (OpenCV format).
    """
    if seed is not None:
        np.random.seed(seed)

    w, h = size_wh
    
    # Skin tone base (varied)
    base_rgb = np.array([
        random.randint(160, 220),
        random.randint(145, 205),
        random.randint(130, 190)
    ], dtype=np.uint8)
    
    base_bgr = base_rgb[::-1]  # Convert to BGR
    
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[:] = base_bgr
    
    # Lighting gradient
    grad = np.linspace(0.92, 1.08, h).reshape(h, 1, 1)
    img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    
    # Face oval
    center = (w // 2, h // 2)
    axes = (max(10, w // 3), max(10, h // 2 - 8))
    face_color = tuple(map(int, np.clip(base_bgr + random.randint(-12, 12), 0, 255)))
    cv2.ellipse(img, center, axes, 0, 0, 360, face_color, -1)
    
    # Eyes
    eye_y = h // 3
    cv2.circle(img, (w // 3, eye_y), max(2, w // 12), (40, 40, 40), -1)
    cv2.circle(img, (2 * w // 3, eye_y), max(2, w // 12), (40, 40, 40), -1)
    
    # Nose
    cv2.line(img, (w // 2, eye_y + 10), (w // 2, eye_y + 40), (100, 90, 80), 2)
    
    # Mouth
    cv2.ellipse(img, (w // 2, eye_y + 60), (max(8, w // 6), max(4, h // 18)), 0, 0, 180, (60, 50, 50), 2)
    
    # Hairline (simple)
    cv2.ellipse(img, (w // 2, h // 6), (w // 3, h // 10), 0, 0, 180, (45, 45, 45), -1)
    
    # ID-photo vignette
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (max(w, h) ** 2)
    vignette = np.clip(1.0 - 1.1 * r2, 0.78, 1.0).astype(np.float32)
    img = np.clip(img.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)
    
    return img
