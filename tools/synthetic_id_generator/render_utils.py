from typing import Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .font_utils import choose_font


def draw_text_return_bbox(
    img_bgr: np.ndarray,
    text: str,
    pos_xy: Tuple[int, int],
    font_size: int = 14,
    color_rgb: Tuple[int, int, int] = (0, 0, 0),
    role: str = "body",
    bold: bool = False,
    italic: bool = False,
    lang_hint: str = "en",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Draw text and return bbox in xyxy format."""
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_path = choose_font(role=role, bold=bold, italic=italic, lang_hint=lang_hint)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text(pos_xy, text, font=font, fill=color_rgb)
    left, top, right, bottom = draw.textbbox(pos_xy, text, font=font)
    bbox = (int(left), int(top), int(right), int(bottom))

    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out, bbox


def draw_microtext(
    img_bgr: np.ndarray,
    text_pattern: str,
    y_pos: int,
    color_rgb: Tuple[int, int, int] = (150, 150, 150),
    font_size: int = 5,
) -> np.ndarray:
    """
    Draw repetitive microtext line (security feature simulation).
    """
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    w, _ = pil_img.size
    
    font_path = choose_font(role="compact", bold=False)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Repeat pattern across width
    repeat_count = int(w / (len(text_pattern) * font_size / 2)) + 2
    full_text = (text_pattern + "   ") * repeat_count
    
    draw.text((0, y_pos), full_text, font=font, fill=color_rgb)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def add_paper_texture(
    img_bgr: np.ndarray, 
    strength: float = 0.15, 
    seed: int | None = None
) -> np.ndarray:
    """Add paper grain texture for print realism."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    h, w = img_bgr.shape[:2]
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    
    # Smooth to simulate paper grain
    noise = cv2.GaussianBlur(noise, (0, 0), 2.0)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    noise = (noise - 0.5) * 2.0  # [-1, 1]

    out = img_bgr.astype(np.float32) * (1.0 + strength * noise[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def add_diagonal_watermark(
    img_bgr: np.ndarray,
    text: str,
    opacity: int = 55,
    angle_deg: float = -18.0,
    font_size: int = 32,
) -> np.ndarray:
    """Add diagonal tiled watermark (SAFETY FEATURE)."""
    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    w, h = base.size

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font_path = choose_font(role="heading", bold=True)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Tile watermark
    step_x = max(240, font_size * 8)
    step_y = max(140, font_size * 4)
    rgba = (20, 20, 20, int(max(0, min(255, opacity))))

    for y in range(-h, h * 2, step_y):
        for x in range(-w, w * 2, step_x):
            draw.text((x, y), text, font=font, fill=rgba)

    layer = layer.rotate(angle_deg, resample=Image.BICUBIC, expand=False)
    out = Image.alpha_composite(base, layer).convert("RGB")
    
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def pack_bbox(xyxy: Tuple[int, int, int, int], width: int, height: int) -> Dict[str, Any]:
    """
    Pack bbox into multiple formats (YOLO-friendly + raw coords).
    """
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1

    # YOLO format: normalized center x, center y, width, height
    x_center_norm = (x1 + w / 2) / width
    y_center_norm = (y1 + h / 2) / height
    w_norm = w / width
    h_norm = h / height

    return {
        "x": float(f"{x_center_norm:.6f}"),
        "y": float(f"{y_center_norm:.6f}"),
        "w": float(f"{w_norm:.6f}"),
        "h": float(f"{h_norm:.6f}"),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }
