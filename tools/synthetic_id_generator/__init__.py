"""
Synthetic ID Card Generator
============================

Generates high-fidelity synthetic PAN and Aadhaar-like training images
with ground truth labels for OCR/KYC model training.

Features:
- Realistic visual styling (bilingual, microtext, QR codes)
- Acquisition realism (scan/phone modes with augmentations)
- Verhoeff checksum validation for Aadhaar
- PAN surname initial matching (5th character)
- YOLO-compatible bounding box annotations
- Safety watermarking (mandatory)

Usage:
    from tools.synthetic_id_generator import (
        GeneratorConfig,
        PANConfig,
        AadhaarConfig,
        PANCardGenerator,
        AadhaarCardGenerator,
    )
    
    config = GeneratorConfig()
    pan_config = PANConfig()
    
    generator = PANCardGenerator(config, pan_config)
    result = generator.generate(status_code="P")
    
    card_image = result["card_image"]  # numpy array (BGR)
    ground_truth = result["ground_truth"]  # dict with fields + bboxes
"""

from .config import GeneratorConfig, PANConfig, AadhaarConfig
from .pan_generator import PANCardGenerator
from .aadhaar_generator import AadhaarCardGenerator
from .utils import (
    generate_pan_number,
    generate_aadhaar_number,
    verify_verhoeff_checksum,
    generate_indian_name,
    generate_dob,
    generate_indian_address,
)

__version__ = "1.0.0"

__all__ = [
    # Configs
    "GeneratorConfig",
    "PANConfig",
    "AadhaarConfig",
    
    # Generators
    "PANCardGenerator",
    "AadhaarCardGenerator",
    
    # Utility functions
    "generate_pan_number",
    "generate_aadhaar_number",
    "verify_verhoeff_checksum",
    "generate_indian_name",
    "generate_dob",
    "generate_indian_address",
]
