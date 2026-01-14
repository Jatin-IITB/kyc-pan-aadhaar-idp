# services/preprocessing/quality.py
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any

# Load config once
# Find project root (3 levels up from services/preprocessing/quality.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "thresholds.yaml"

try:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f) or {}
            Q_CFG = config.get("quality_gate", {})
    else:
        Q_CFG = {}
except Exception:
    Q_CFG = {}

# DEFAULTS
MIN_BLUR = float(Q_CFG.get("min_blur_score", 35.0))
MAX_WHITE = float(Q_CFG.get("max_white_ratio", 0.50))
MAX_BLACK = float(Q_CFG.get("max_black_ratio", 0.60))
MAX_RES = int(Q_CFG.get("max_resolution", 1600))

def resize_if_huge(img: np.ndarray, max_dim: int = MAX_RES) -> np.ndarray:
    if img is None: return img
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

def check_image_quality(img_bgr: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    """
    Analyzes image quality.
    Returns: (passed_hard_gate, metrics)
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, {"error": "Empty image"}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = blur_score < MIN_BLUR

    # 2. Exposure
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    
    overexposed_pixels = np.sum(hist[250:])
    underexposed_pixels = np.sum(hist[:5])
    
    overexposed_ratio = overexposed_pixels / total_pixels
    underexposed_ratio = underexposed_pixels / total_pixels
    
    is_overexposed = overexposed_ratio > MAX_WHITE
    is_underexposed = underexposed_ratio > MAX_BLACK

    metrics = {
        "blur_score": float(blur_score),
        "overexposed_ratio": float(overexposed_ratio),
        "underexposed_ratio": float(underexposed_ratio),
        "is_blurry": bool(is_blurry),
        "is_overexposed": bool(is_overexposed),
        "is_underexposed": bool(is_underexposed),
        "rejection_reason": None
    }

    # Initial Decision
    if is_blurry:
        metrics["rejection_reason"] = f"Image is too blurry (Score: {blur_score:.1f} < {MIN_BLUR})"
        return False, metrics
    
    if is_overexposed:
        metrics["rejection_reason"] = f"Image is overexposed (White Ratio: {overexposed_ratio:.2f} > {MAX_WHITE})"
        return False, metrics
        
    if is_underexposed:
        metrics["rejection_reason"] = f"Image is too dark (Dark Ratio: {underexposed_ratio:.2f} > {MAX_BLACK})"
        return False, metrics

    return True, metrics
