import numpy as np
import cv2
from typing import Dict, Any

from .config import GeneratorConfig, PANConfig
from .utils import generate_pan_number, generate_indian_name, generate_dob, generate_face_photo, choose_style
from .render_utils import (
    draw_text_return_bbox,
    pack_bbox,
    draw_microtext,
    add_paper_texture,
    add_diagonal_watermark,
)
from .augmentations import apply_augmentations
from .tempalate_utils import load_template_or_none


class PANCardGenerator:
    """PAN-like synthetic card generator with high realism."""

    def __init__(self, config: GeneratorConfig, pan_config: PANConfig):
        self.config = config
        self.pan_config = pan_config

    def _create_card_template(self) -> np.ndarray:
        """Create base PAN card background."""
        w, h = self.config.card_size
        img = np.ones((h, w, 3), dtype=np.uint8)
        img[:] = self.pan_config.template_color

        # Subtle gradient
        grad = np.linspace(1.03, 0.97, h).astype(np.float32)[:, None, None]
        img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)

        # Border
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (150, 150, 150), 2)
        
        return img

    def generate(self, status_code: str = "P", output_id: str | None = None) -> Dict[str, Any]:
        # Generate data
        name, father_name = generate_indian_name()
        # CRITICAL: Pass name to ensure 5th char = surname initial
        pan_number = generate_pan_number(status_code, name=name)
        dob = generate_dob()

        w, h = self.config.card_size

        # Load/Create template
        template_path = str(self.config.templates_dir / self.config.pan_template_name)
        card_img = load_template_or_none(template_path, w, h)
        if card_img is None:
            card_img = self._create_card_template()

        bboxes: Dict[str, Any] = {}

        # --- Header Band ---
        cv2.rectangle(card_img, (10, 10), (w - 10, 65), self.pan_config.header_color, -1)
        
        header_text = "INCOME TAX DEPARTMENT"
        card_img, bb = draw_text_return_bbox(
            card_img, header_text, (0, 40),
            font_size=24, color_rgb=(255, 255, 255), role="heading", bold=True
        )
        # Center text
        text_w = bb[2] - bb[0]
        card_img, bb = draw_text_return_bbox(
            card_img, header_text, ((w - text_w) // 2, 40),
            font_size=24, color_rgb=(255, 255, 255), role="heading", bold=True
        )
        bboxes["header_text"] = pack_bbox(bb, w, h)

        # --- Microtext Security ---
        microtext_pattern = f"INCOME TAX INDIA {pan_number} "
        card_img = draw_microtext(card_img, microtext_pattern, y_pos=75, color_rgb=(100, 100, 100))

        # --- Photo (Top Right) ---
        px_w, px_h = self.pan_config.photo_size
        photo_x1, photo_y1 = w - px_w - 40, 120
        photo_x2, photo_y2 = photo_x1 + px_w, photo_y1 + px_h
        
        photo_img = generate_face_photo((px_w, px_h))
        card_img[photo_y1:photo_y2, photo_x1:photo_x2] = photo_img
        cv2.rectangle(card_img, (photo_x1, photo_y1), (photo_x2, photo_y2), (0, 0, 0), 2)
        bboxes["photo"] = pack_bbox((photo_x1, photo_y1, photo_x2, photo_y2), w, h)

        # --- Data Fields ---
        y_cursor = 120
        x_start = 40
        line_height = 50

        # PAN Number
        label = "Permanent Account Number"
        card_img, bb = draw_text_return_bbox(card_img, label, (x_start, y_cursor), font_size=14, bold=True)
        y_cursor += 22
        card_img, bb = draw_text_return_bbox(card_img, pan_number, (x_start, y_cursor), font_size=20, role="compact", bold=True)
        bboxes["pan_number"] = pack_bbox(bb, w, h)
        y_cursor += line_height

        # Name
        label = "Name / नाम"
        card_img, bb = draw_text_return_bbox(card_img, label, (x_start, y_cursor), font_size=14, bold=True)
        y_cursor += 22
        card_img, bb = draw_text_return_bbox(card_img, name, (x_start, y_cursor), font_size=16, bold=True)
        bboxes["name"] = pack_bbox(bb, w, h)
        y_cursor += line_height

        # Father's Name
        label = "Father's Name / पिता का नाम"
        card_img, bb = draw_text_return_bbox(card_img, label, (x_start, y_cursor), font_size=14, bold=True)
        y_cursor += 22
        card_img, bb = draw_text_return_bbox(card_img, father_name, (x_start, y_cursor), font_size=16)
        bboxes["father_name"] = pack_bbox(bb, w, h)
        y_cursor += line_height

        # Date of Birth
        label = "Date of Birth / जन्म तिथि"
        card_img, bb = draw_text_return_bbox(card_img, label, (x_start, y_cursor), font_size=14, bold=True)
        y_cursor += 22
        card_img, bb = draw_text_return_bbox(card_img, dob, (x_start, y_cursor), font_size=16, role="compact", bold=True)
        bboxes["date_of_birth"] = pack_bbox(bb, w, h)

        # Signature Line
        sign_x, sign_y = photo_x1, photo_y2 + 25
        cv2.line(card_img, (sign_x, sign_y), (sign_x + 100, sign_y), (0, 0, 0), 1)
        card_img, bb = draw_text_return_bbox(card_img, "Signature", (sign_x, sign_y + 5), font_size=12, color_rgb=(50, 50, 50))
        bboxes["signature_label"] = pack_bbox(bb, w, h)

        # --- Paper Texture + Watermark (BEFORE augmentations) ---
        card_img = add_paper_texture(card_img, strength=0.12)
        # card_img = add_diagonal_watermark(
        #     card_img,
        #     text="TESTING KYC • SYNTHETIC",
        #     opacity=50,
        #     angle_deg=-18.0,
        #     font_size=28,
        # )

        # --- Acquisition Augmentations ---
        style = choose_style(self.config.style_weights)
        if self.config.enable_augmentation:
            card_img = apply_augmentations(card_img, mode=style)

        ground_truth = {
            "doc_type": "PAN",
            "acquisition_style": style,
            "fields": {
                "pan_number": pan_number,
                "name": name,
                "father_name": father_name,
                "date_of_birth": dob,
                "status_code": status_code,
                "status_meaning": self.pan_config.status_codes.get(status_code, "Unknown"),
            },
            "bboxes": bboxes,
            "card_size": {"w": w, "h": h},
        }

        return {
            "card_image": card_img,
            "ground_truth": ground_truth,
            "doc_id": output_id or f"PAN_{pan_number}",
        }
