"""
Font selection utilities for multilingual text rendering.
Supports Windows system fonts + bundled Noto fonts for Hindi/Devanagari.
"""

import platform
from pathlib import Path


# --- Windows System Fonts ---
WINDOWS_FONTS = {
    "arial": r"C:\Windows\Fonts\arial.ttf",
    "arial_bold": r"C:\Windows\Fonts\arialbd.ttf",
    "arial_italic": r"C:\Windows\Fonts\ariali.ttf",
    "arial_bold_italic": r"C:\Windows\Fonts\arialbi.ttf",
    
    # Arial Narrow (compact variant)
    "arial_narrow": r"C:\Windows\Fonts\arialn.ttf",
    "arial_narrow_bold": r"C:\Windows\Fonts\arialnb.ttf",
    "arial_narrow_italic": r"C:\Windows\Fonts\arialni.ttf",
    "arial_narrow_bold_italic": r"C:\Windows\Fonts\arialnbi.ttf",
    
    # Hindi/Devanagari support
    "mangal": r"C:\Windows\Fonts\mangal.ttf",           # Standard Hindi font
    "nirmala": r"C:\Windows\Fonts\Nirmala.ttf",         # Modern Hindi font
    "kokila": r"C:\Windows\Fonts\kokila.ttf",           # Serif Hindi font
}


# --- Bundled Fonts (cross-platform, recommended) ---
# Place these in: data/fonts/
BUNDLED_FONTS = {
    "noto_sans": Path("data/fonts/NotoSans-Regular.ttf"),
    "noto_sans_bold": Path("data/fonts/NotoSans-Bold.ttf"),
    "noto_devanagari": Path("data/fonts/NotoSansDevanagari-Regular.ttf"),
    "noto_devanagari_bold": Path("data/fonts/NotoSansDevanagari-Bold.ttf"),
}


def _path_exists(p: str | Path) -> bool:
    """Check if a file path exists."""
    return Path(p).exists()


def _project_root() -> Path:
    """Get project root (3 levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def _resolve_bundled(rel_path: Path) -> Path:
    """Resolve bundled font path relative to project root."""
    return _project_root() / rel_path


def choose_font(
    role: str = "body",
    bold: bool = False,
    italic: bool = False,
    lang_hint: str = "en"
) -> str:
    """
    Select appropriate font based on role and language.
    
    Args:
        role: Font role - "body", "heading", "compact", "hindi"
        bold: Whether to use bold variant
        italic: Whether to use italic variant
        lang_hint: Language hint - "en" (English) or "hi" (Hindi/Devanagari)
    
    Returns:
        Absolute path to font file
    
    Priority:
        1. Bundled fonts (cross-platform, recommended)
        2. Windows system fonts (fallback)
        3. Default Arial (last resort)
    """
    system = platform.system()
    is_hindi = lang_hint.startswith("hi") or role == "hindi"
    
    # --- Hindi/Devanagari Fonts ---
    if is_hindi:
        # Try bundled Noto Devanagari first
        if bold:
            bundled = _resolve_bundled(BUNDLED_FONTS["noto_devanagari_bold"])
            if _path_exists(bundled):
                return str(bundled)
        
        bundled = _resolve_bundled(BUNDLED_FONTS["noto_devanagari"])
        if _path_exists(bundled):
            return str(bundled)
        
        # Fallback to Windows Hindi fonts
        if system == "Windows":
            for font_name in ["nirmala", "mangal", "kokila"]:
                if _path_exists(WINDOWS_FONTS[font_name]):
                    return WINDOWS_FONTS[font_name]
        
        # Last resort: regular Arial (will not render Hindi properly)
        return WINDOWS_FONTS["arial"]
    
    # --- English Fonts ---
    
    # Try bundled Noto Sans first (cross-platform)
    if bold:
        bundled = _resolve_bundled(BUNDLED_FONTS["noto_sans_bold"])
        if _path_exists(bundled):
            return str(bundled)
    
    bundled = _resolve_bundled(BUNDLED_FONTS["noto_sans"])
    if _path_exists(bundled):
        return str(bundled)
    
    # Fallback to Windows Arial variants
    if system != "Windows":
        return WINDOWS_FONTS["arial"]  # Will fail gracefully on non-Windows
    
    # Role-specific Windows font selection
    if role == "heading":
        return WINDOWS_FONTS["arial_bold"] if _path_exists(WINDOWS_FONTS["arial_bold"]) else WINDOWS_FONTS["arial"]
    
    if role == "compact":
        # Arial Narrow variants
        if bold and italic and _path_exists(WINDOWS_FONTS["arial_narrow_bold_italic"]):
            return WINDOWS_FONTS["arial_narrow_bold_italic"]
        if bold and _path_exists(WINDOWS_FONTS["arial_narrow_bold"]):
            return WINDOWS_FONTS["arial_narrow_bold"]
        if italic and _path_exists(WINDOWS_FONTS["arial_narrow_italic"]):
            return WINDOWS_FONTS["arial_narrow_italic"]
        if _path_exists(WINDOWS_FONTS["arial_narrow"]):
            return WINDOWS_FONTS["arial_narrow"]
        return WINDOWS_FONTS["arial"]
    
    # Body text (default)
    if bold and italic and _path_exists(WINDOWS_FONTS["arial_bold_italic"]):
        return WINDOWS_FONTS["arial_bold_italic"]
    if bold and _path_exists(WINDOWS_FONTS["arial_bold"]):
        return WINDOWS_FONTS["arial_bold"]
    if italic and _path_exists(WINDOWS_FONTS["arial_italic"]):
        return WINDOWS_FONTS["arial_italic"]
    
    return WINDOWS_FONTS["arial"]
