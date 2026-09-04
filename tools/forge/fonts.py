"""Cross-platform font resolution for document rendering."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator

from PIL import ImageFont

_OVERRIDE: Dict[str, str] = {}


@contextmanager
def font_override(mapping: Dict[str, str]) -> Iterator[None]:
    """Temporarily remap font kinds (e.g. bold -> serif) for tamper synthesis."""
    global _OVERRIDE
    prev = _OVERRIDE
    _OVERRIDE = {**prev, **mapping}
    try:
        yield
    finally:
        _OVERRIDE = prev

_CANDIDATES = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\verdanab.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "mono": [
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        r"C:\Windows\Fonts\courbd.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ],
    "serif": [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ],
}

KINDS = ("regular", "bold", "mono", "serif")


def resolve(kind: str) -> str | None:
    """First existing candidate path for *kind*, or None if none resolve."""
    for path in _CANDIDATES.get(kind, []):
        if Path(path).exists():
            return path
    return None


def font_environment() -> Dict[str, str | None]:
    """Which concrete font file each kind resolves to on this machine."""
    return {k: resolve(k) for k in KINDS}


def check_font_environment() -> list:
    """Problems that would silently invalidate every rendered document.

    Two failure modes produce garbage without raising:

    * A kind resolving to nothing falls back to PIL's default bitmap font.
    * Two kinds resolving to the SAME file make font_swap a no-op — the
      "attack" renders a byte-identical card.

    Either way the typographic envelope in ``config/font_profiles.json``,
    fit on a machine where all four resolved to distinct typefaces, no longer
    describes the corpus. Measured on Windows, where _CANDIDATES previously
    had no entries at all: 83% of genuine documents breached the envelope,
    giving 86.7% genuine FPR, while font_swap "recall" rose to 0.83 purely
    because the detector was flagging everything.
    """
    env = font_environment()
    problems = []
    missing = sorted(k for k, v in env.items() if v is None)
    if missing:
        problems.append(
            f"no font file found for: {', '.join(missing)} — these fall back to "
            f"PIL's default bitmap font, so rendered typography is meaningless")
    seen: Dict[str, list] = {}
    for kind, path in env.items():
        if path:
            seen.setdefault(path, []).append(kind)
    for path, kinds in sorted(seen.items()):
        if len(kinds) > 1:
            problems.append(
                f"{', '.join(sorted(kinds))} all resolve to {path} — font_swap "
                f"cannot alter typography")
    return problems


def load_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return _load(_OVERRIDE.get(kind, kind), size)


@lru_cache(maxsize=64)
def _load(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _CANDIDATES.get(kind, []):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
