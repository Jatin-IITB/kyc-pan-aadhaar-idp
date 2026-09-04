"""Guards on the rendering font environment.

The forge renders four distinct typographic kinds. If any kind cannot be
resolved to a real font file, PIL silently substitutes its default bitmap
font; if two kinds resolve to the same file, font_swap becomes a no-op. Both
failures are invisible at render time and invalidate every downstream number:
measured on Windows (where _CANDIDATES carried no Windows paths at all),
83% of genuine documents breached the calibrated envelope, producing 86.7%
genuine FPR while font_swap "recall" rose to 0.83 because the detector was
flagging everything.
"""
import pytest

from tools.forge import fonts


def test_all_kinds_resolve_on_this_platform():
    env = fonts.font_environment()
    missing = sorted(k for k, v in env.items() if v is None)
    assert not missing, (
        f"no font file resolved for {missing}; these would fall back to PIL's "
        f"default bitmap font. Resolved: {env}")


def test_kinds_resolve_to_distinct_files():
    env = fonts.font_environment()
    resolved = [v for v in env.values() if v]
    assert len(set(resolved)) == len(resolved), (
        f"font kinds collapsed onto the same file, font_swap would be a "
        f"no-op: {env}")


def test_check_font_environment_is_clean_here():
    assert fonts.check_font_environment() == []


def test_candidates_cover_macos_linux_and_windows():
    """A platform with no candidate paths silently degrades to bitmap text."""
    for kind in fonts.KINDS:
        paths = fonts._CANDIDATES[kind]
        assert any(p.startswith("/System/") for p in paths), f"{kind}: no macOS path"
        assert any(p.startswith("/usr/share/") for p in paths), f"{kind}: no Linux path"
        assert any(p[1:3] == ":\\" for p in paths), f"{kind}: no Windows path"


def test_check_reports_missing_kind(monkeypatch):
    monkeypatch.setitem(fonts._CANDIDATES, "serif", ["/nonexistent/none.ttf"])
    problems = fonts.check_font_environment()
    assert any("serif" in p for p in problems)


def test_check_reports_collapsed_kinds(monkeypatch):
    """Two kinds pointing at one file must be reported, not tolerated."""
    shared = fonts.resolve("regular")
    if shared is None:
        pytest.skip("no resolvable font on this platform")
    monkeypatch.setitem(fonts._CANDIDATES, "serif", [shared])
    problems = fonts.check_font_environment()
    assert any("font_swap" in p for p in problems)
