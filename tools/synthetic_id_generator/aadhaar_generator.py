import numpy as np
import cv2
from typing import Dict, Any
import random

from .config import GeneratorConfig, AadhaarConfig
from .utils import (
    generate_aadhaar_number,
    verify_verhoeff_checksum,
    generate_indian_name,
    generate_dob,
    generate_indian_address,
    generate_face_photo,
    choose_style,
)
from .render_utils import draw_text_return_bbox, pack_bbox, draw_microtext
from .augmentations import apply_augmentations
from .qr_utils import generate_qr_image, paste_rgb_roi
from .tempalate_utils import load_template_or_none


class AadhaarCardGenerator:
    """
    Aadhaar-like synthetic card generator with high visual fidelity.
    Features: QR code, bilingual labels, microtext security, address wrapping.
    """

    def __init__(self, config: GeneratorConfig, aadhaar_config: AadhaarConfig):
        self.config = config
        self.aadhaar_config = aadhaar_config

    def _create_card_template(self) -> np.ndarray:
        """Create base Aadhaar-like card background."""
        w, h = self.config.card_size
        img = np.ones((h, w, 3), dtype=np.uint8)
        img[:] = self.aadhaar_config.template_color  # Light gray/white

        # Subtle gradient for realism
        grad_x = np.linspace(1.02, 0.98, w).astype(np.float32)
        grad_y = np.linspace(1.01, 0.99, h).astype(np.float32)
        grad_2d = np.outer(grad_y, grad_x)
        img = np.clip(img.astype(np.float32) * grad_2d[..., None], 0, 255).astype(np.uint8)

        # Border
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (150, 150, 150), 2)

        # Top accent bar (purple-ish)
        accent_bgr = self.aadhaar_config.accent_color
        cv2.rectangle(img, (10, 10), (w - 10, 70), accent_bgr, -1)

        return img

    def generate(self, output_id: str | None = None) -> Dict[str, Any]:
        # Generate data
        aadhaar_number = generate_aadhaar_number()
        assert verify_verhoeff_checksum(aadhaar_number), "Invalid Aadhaar checksum!"

        gender = random.choice(["Male", "Female"])
        name, _ = generate_indian_name(gender)
        dob = generate_dob()
        address = generate_indian_address()
        formatted_aadhaar = f"{aadhaar_number[:4]} {aadhaar_number[4:8]} {aadhaar_number[8:]}"

        w, h = self.config.card_size

        # Load/Create card
        template_path = str(self.config.templates_dir / self.config.aadhaar_template_name)
        card_img = load_template_or_none(template_path, w, h)
        if card_img is None:
            card_img = self._create_card_template()

        bboxes: Dict[str, Any] = {}
        text_rgb = self.aadhaar_config.text_color[::-1]  # BGR -> RGB

        # --- 1. Header (Bilingual) ---
        header_en = "UNIQUE IDENTIFICATION AUTHORITY OF INDIA"
        header_hi = "भारतीय विशिष्ट पहचान प्राधिकरण"

        # English header
        card_img, bb_en = draw_text_return_bbox(
            card_img, header_en, (w // 2, 25),
            font_size=18, color_rgb=(255, 255, 255), role="heading", bold=True
        )
        # Re-center
        text_w = bb_en[2] - bb_en[0]
        card_img, bb_en = draw_text_return_bbox(
            card_img, header_en, ((w - text_w) // 2, 25),
            font_size=18, color_rgb=(255, 255, 255), role="heading", bold=True
        )

        # Hindi header
        card_img, bb_hi = draw_text_return_bbox(
            card_img, header_hi, (w // 2, 48),
            font_size=14, color_rgb=(255, 255, 255), role="hindi", bold=True, lang_hint="hi"
        )
        text_w_hi = bb_hi[2] - bb_hi[0]
        card_img, bb_hi = draw_text_return_bbox(
            card_img, header_hi, ((w - text_w_hi) // 2, 48),
            font_size=14, color_rgb=(255, 255, 255), role="hindi", bold=True, lang_hint="hi"
        )

        # Combined header bbox
        x1 = min(bb_en[0], bb_hi[0])
        y1 = min(bb_en[1], bb_hi[1])
        x2 = max(bb_en[2], bb_hi[2])
        y2 = max(bb_en[3], bb_hi[3])
        bboxes["header"] = pack_bbox((x1, y1, x2, y2), w, h)

        # --- 2. Microtext Security Feature ---
        microtext_pattern = f"AADHAAR {formatted_aadhaar} UIDAI "
        card_img = draw_microtext(card_img, microtext_pattern, y_pos=80, color_rgb=(120, 120, 120))

        # --- 3. Photo (Left Side) ---
        px_w, px_h = self.aadhaar_config.photo_size
        photo_x1, photo_y1 = 35, 110
        photo_x2, photo_y2 = photo_x1 + px_w, photo_y1 + px_h

        photo_img = generate_face_photo((px_w, px_h))
        card_img[photo_y1:photo_y2, photo_x1:photo_x2] = photo_img
        cv2.rectangle(card_img, (photo_x1, photo_y1), (photo_x2, photo_y2), (30, 30, 30), 2)
        bboxes["photo"] = pack_bbox((photo_x1, photo_y1, photo_x2, photo_y2), w, h)

        # --- 4. Data Fields (Right of Photo) ---
        x_start = photo_x2 + 30
        y_cursor = 110
        line_height = 50

        # Name
        label_en = "Name / नाम"
        card_img, bb = draw_text_return_bbox(card_img, label_en, (x_start, y_cursor), font_size=12, bold=True)
        y_cursor += 22
        card_img, bb = draw_text_return_bbox(card_img, name, (x_start, y_cursor), font_size=18, bold=True)
        bboxes["name"] = pack_bbox(bb, w, h)
        y_cursor += line_height

        # DOB / Gender on same line
        label_dob = "DOB / जन्म तिथि"
        card_img, bb = draw_text_return_bbox(card_img, label_dob, (x_start, y_cursor), font_size=11, bold=True)
        y_cursor += 20
        card_img, bb_dob = draw_text_return_bbox(card_img, dob, (x_start, y_cursor), font_size=14, role="compact", bold=True)
        bboxes["date_of_birth"] = pack_bbox(bb_dob, w, h)

        # Gender (to the right of DOB)
        gender_x = x_start + 200
        label_gender = "Gender / लिंग"
        card_img, bb = draw_text_return_bbox(card_img, label_gender, (gender_x, y_cursor - 20), font_size=11, bold=True)
        card_img, bb_gender = draw_text_return_bbox(card_img, gender, (gender_x, y_cursor), font_size=14, bold=True)
        bboxes["gender"] = pack_bbox(bb_gender, w, h)

        y_cursor += line_height

        # --- 5. Address (Wrapped) ---
        label_address = "Address / पता"
        card_img, bb = draw_text_return_bbox(card_img, label_address, (35, y_cursor), font_size=12, bold=True)
        y_cursor += 22

        # Wrap address into multiple lines
        words = address.split()
        lines = []
        current_line = []
        max_chars_per_line = 52

        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) > max_chars_per_line:
                if len(current_line) > 1:
                    lines.append(" ".join(current_line[:-1]))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []

        if current_line:
            lines.append(" ".join(current_line))

        addr_bboxes = []
        for i, line in enumerate(lines[:3]):  # Max 3 lines
            card_img, bb_line = draw_text_return_bbox(
                card_img, line, (35, y_cursor + i * 18), font_size=12
            )
            addr_bboxes.append(bb_line)

        if addr_bboxes:
            ax1 = min(b[0] for b in addr_bboxes)
            ay1 = min(b[1] for b in addr_bboxes)
            ax2 = max(b[2] for b in addr_bboxes)
            ay2 = max(b[3] for b in addr_bboxes)
            bboxes["address"] = pack_bbox((ax1, ay1, ax2, ay2), w, h)

        # --- 6. Aadhaar Number (Bottom Center, Prominent) ---
        aadhaar_y = h - 80
        card_img, bb = draw_text_return_bbox(
            card_img, formatted_aadhaar, (w // 2, aadhaar_y),
            font_size=32, role="compact", bold=True, color_rgb=(138, 43, 226)
        )
        # Re-center
        text_w_a = bb[2] - bb[0]
        card_img, bb = draw_text_return_bbox(
            card_img, formatted_aadhaar, ((w - text_w_a) // 2, aadhaar_y),
            font_size=32, role="compact", bold=True, color_rgb=(138, 43, 226)
        )
        bboxes["aadhaar_number"] = pack_bbox(bb, w, h)

        # --- 7. QR Code (Bottom Right) ---
        qr_size = self.aadhaar_config.qr_code_size
        qr_x1 = w - qr_size - 40
        qr_y1 = h - qr_size - 40
        qr_x2 = qr_x1 + qr_size
        qr_y2 = qr_y1 + qr_size

        qr_payload = f"uid:{aadhaar_number}|name:{name}|dob:{dob}|gender:{gender}"
        qr_rgb = generate_qr_image(qr_payload, size=qr_size)
        card_img = paste_rgb_roi(card_img, qr_rgb, (qr_x1, qr_y1, qr_x2, qr_y2))
        cv2.rectangle(card_img, (qr_x1, qr_y1), (qr_x2, qr_y2), (30, 30, 30), 1)
        bboxes["qr_code"] = pack_bbox((qr_x1, qr_y1, qr_x2, qr_y2), w, h)

        # --- 8. Augmentations ---
        style = choose_style(self.config.style_weights)
        if self.config.enable_augmentation:
            card_img = apply_augmentations(card_img, mode=style)

        ground_truth = {
            "doc_type": "Aadhaar",
            "acquisition_style": style,
            "fields": {
                "aadhaar_number": aadhaar_number,
                "aadhaar_formatted": formatted_aadhaar,
                "name": name,
                "date_of_birth": dob,
                "gender": gender,
                "address": address,
                "checksum_valid": True,
                "qr_payload": qr_payload,
            },
            "bboxes": bboxes,
            "card_size": {"w": w, "h": h},
        }

        return {
            "card_image": card_img,
            "ground_truth": ground_truth,
            "doc_id": output_id or f"Aadhaar_{aadhaar_number}",
        }
