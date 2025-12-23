"""Configuration dataclasses for synthetic ID generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class GeneratorConfig:
    """Main configuration for document generation pipeline."""
    
    # Output paths
    output_dir: Path = Path("data/synthetic")
    images_dirname: str = "images"
    labels_dirname: str = "labels"
    manifest_name: str = "manifest.csv"
    
    # Card dimensions (standard ID card aspect ratio)
    card_size: Tuple[int, int] = (856, 540)  # (Width, Height) in pixels
    dpi: int = 300
    
    # Template images (optional; falls back to programmatic generation)
    templates_dir: Path = Path("data/templates")
    pan_template_name: str = "pan_base.png"
    aadhaar_template_name: str = "aadhaar_base.png"
    
    # Watermark (SAFETY: always enabled for synthetic data)
    watermark_text: str = "TESTING KYC • SYNTHETIC • NOT REAL"
    watermark_opacity: int = 50  # 0-255 (alpha channel)
    watermark_angle_deg: float = -18.0
    
    # Acquisition styles (weights for random sampling)
    enable_augmentation: bool = True
    style_weights: Tuple[float, float] = (0.35, 0.65)  # (scan_weight, phone_weight)
    
    # Future: Full scene generation (card on desk/background)
    generate_full_scene: bool = False
    scene_size: Tuple[int, int] = (1920, 1080)
    background_colors: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (240, 240, 240),  # Light gray
            (255, 255, 255),  # White
            (220, 220, 210),  # Beige
            (200, 180, 160),  # Tan
        ]
    )


@dataclass
class PANConfig:
    """PAN card visual styling configuration."""
    
    # Colors (BGR format for OpenCV)
    template_color: Tuple[int, int, int] = (235, 245, 255)  # Light cream
    header_color: Tuple[int, int, int] = (112, 25, 25)      # Navy blue
    text_color: Tuple[int, int, int] = (0, 0, 0)            # Black
    
    # Layout dimensions
    photo_size: Tuple[int, int] = (110, 135)  # (Width, Height) in pixels
    
    # PAN status codes (4th character meanings)
    status_codes: dict = field(
        default_factory=lambda: {
            "P": "Individual/Person",
            "C": "Company",
            "H": "Hindu Undivided Family (HUF)",
            "F": "Firm/Partnership",
            "A": "Association of Persons (AOP)",
            "T": "Trust",
            "B": "Body of Individuals (BOI)",
            "L": "Local Authority",
            "J": "Artificial Juridical Person",
            "G": "Government",
        }
    )


@dataclass
class AadhaarConfig:
    """Aadhaar card visual styling configuration."""
    
    # Colors (BGR format for OpenCV)
    template_color: Tuple[int, int, int] = (245, 245, 245)  # Off-white
    accent_color: Tuple[int, int, int] = (226, 43, 138)     # Purple accent
    text_color: Tuple[int, int, int] = (0, 0, 0)            # Black
    
    # Layout dimensions
    photo_size: Tuple[int, int] = (125, 150)  # (Width, Height) in pixels
    qr_code_size: int = 150  # Square QR code size
