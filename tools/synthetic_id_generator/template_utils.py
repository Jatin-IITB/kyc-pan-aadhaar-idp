from pathlib import Path
import cv2
import numpy as np


def project_root() -> Path:
    """
    Get the project root directory (3 levels up from this file).
    Adjust if your file structure differs.
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path) -> Path:
    """
    Resolve a path: if absolute, use as-is; if relative, resolve from project root.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return project_root() / p


def load_template_or_none(path: str | Path, width: int, height: int) -> np.ndarray | None:
    """
    Load a template image from the given path and resize it.
    
    Returns None if the file doesn't exist or cannot be loaded.
    Useful for fallback to programmatically generated templates.
    
    Args:
        path: Path to template image (can be relative to project root)
        width: Target width
        height: Target height
    
    Returns:
        BGR numpy array or None
    """
    resolved_path = resolve_path(path)
    
    if not resolved_path.exists():
        return None
    
    img = cv2.imread(str(resolved_path))
    
    if img is None:
        return None
    
    # Resize to target dimensions
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    return img
