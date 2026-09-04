"""Measure forensic recall as a function of the attacker's JPEG save quality.

ADR-028 flagged `text_splice` and `regenerate` recall as tautological: both
attacks are detected by the `low_jpeg_quality` metadata gate, which fires on
JPEG quality < 88 — an artifact of the save quality the tamper forge happens
to use, not evidence of tampering. The README has carried that caveat as an
assertion ("trivially evadable by a competent adversary saving at Q>=88").

This tool turns the assertion into a measurement. It re-runs both attacks
across a sweep of attacker-chosen save qualities, holding everything else
fixed, and reports end-to-end recall through the real detector stack and
spoof scorer. The cliff at the 88 threshold is the tautology, drawn.

    .venv/bin/python -m tools.eval.evasion_probe
    .venv/bin/python -m tools.eval.evasion_probe --qualities 92,95 --split tuning

Only the JPEG quality parameter is varied: cv2.imencode is intercepted inside
tamper_forge so the painted splice, the re-render, and the double-compression
loop are all byte-identical to a normal forge run.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DOC_TYPES = ("pan", "aadhaar", "driving_license")
SEVERITIES = ("low", "med", "high")
EVADABLE_ATTACKS = ("text_splice", "regenerate")
DEFAULT_QUALITIES = (70, 78, 85, 88, 90, 92, 95, 98)


@contextlib.contextmanager
def forced_jpeg_quality(q: int):
    """Force every cv2.imencode('.jpg', ...) inside tamper_forge to quality q."""
    import tools.forge.tamper_forge as tf

    real = cv2.imencode

    def patched(ext, img, params=None):
        if ext == ".jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, q]
        return real(ext, img, params) if params is not None else real(ext, img)

    tf.cv2.imencode = patched
    try:
        yield
    finally:
        tf.cv2.imencode = real


def _build_stack():
    from services.forensics.copy_move import CopyMoveDetector
    from services.forensics.ela import ELADetector
    from services.forensics.font_analysis import FontConsistencyAnalyzer
    from services.forensics.font_profile import TemplateFontForensics
    from services.forensics.metadata import MetadataForensics
    from services.forensics.screen_recapture import ScreenRecaptureDetector
    from services.forensics.spoof_scorer import SpoofScorer

    return ({
        "ela": ELADetector(), "copy_move": CopyMoveDetector(),
        "font": FontConsistencyAnalyzer(), "metadata": MetadataForensics(),
        "screen": ScreenRecaptureDetector(),
    }, TemplateFontForensics(), SpoofScorer())


def _sweep(img, raw, doc_type, detectors, template_font, scorer):
    """Mirror of run_eval.run_forensics.sweep — same detectors, same scorer."""
    r = {}
    for name, det in detectors.items():
        if name == "metadata":
            r[name] = det.analyze(raw)
        elif name == "ela":
            r[name] = det.analyze(img)
        elif name == "font":
            r[name] = det.analyze(img, [])
        else:
            r[name] = det.detect(img)
    ft = template_font.analyze(img, doc_type)
    return scorer.compute(r["ela"], r["copy_move"], r["font"], r["metadata"],
                          r["screen"], font_template_result=ft), r["metadata"]


def run(split: str, qualities) -> dict:
    from tools.forge.tamper_forge import attack_regenerate, attack_text_splice

    detectors, template_font, scorer = _build_stack()
    cases = []
    for dt in DOC_TYPES:
        tdir = REPO / "data" / split / "synthetic" / dt / "truth"
        idir = REPO / "data" / split / "synthetic" / dt / "images"
        for tp in sorted(tdir.glob("*.json")):
            img = cv2.imread(str(idir / f"{tp.stem}.jpg"))
            if img is not None:
                cases.append((dt, tp.stem, json.loads(tp.read_text()), img))
    if not cases:
        raise SystemExit(f"no genuine images under data/{split}/synthetic — run `make forge`")

    out = {}
    for q in qualities:
        per = {a: {"flagged": 0, "n": 0, "lowq": 0, "spoof": []} for a in EVADABLE_ATTACKS}
        for dt, stem, truth, img in cases:
            for attack in EVADABLE_ATTACKS:
                for sev in SEVERITIES:
                    rng = np.random.default_rng(abs(hash((stem, attack, sev))) % 2**32)
                    with forced_jpeg_quality(q):
                        if attack == "text_splice":
                            forged, _lbl, data = attack_text_splice(img, truth, rng, sev)
                        else:
                            forged, _lbl, data = attack_regenerate(img, truth, rng, sev, dt)
                    res, meta = _sweep(forged, data, dt, detectors, template_font, scorer)
                    p = per[attack]
                    p["n"] += 1
                    p["flagged"] += res["recommendation"] != "PASS"
                    p["lowq"] += "low_jpeg_quality" in meta["metadata_flags"]
                    p["spoof"].append(res["spoof_score"])
        out[q] = {a: {"recall": v["flagged"] / max(v["n"], 1),
                      "n": v["n"],
                      "lowq_rate": v["lowq"] / max(v["n"], 1),
                      "mean_spoof": float(np.mean(v["spoof"]))}
                  for a, v in per.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout", choices=("holdout", "tuning"))
    ap.add_argument("--qualities", default=",".join(str(q) for q in DEFAULT_QUALITIES))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    qualities = [int(q) for q in args.qualities.split(",")]
    res = run(args.split, qualities)

    from services.forensics.metadata import MetadataForensics
    thr = MetadataForensics.LOW_QUALITY_THRESHOLD

    print(f"\nEvasion curve — {args.split} split, {len(DOC_TYPES)} doc types x "
          f"{len(SEVERITIES)} severities")
    print(f"low_jpeg_quality gate fires below Q={thr}\n")
    print(f"{'save Q':>7s} | {'text_splice':>22s} | {'regenerate':>22s}")
    print(f"{'':>7s} | {'recall':>8s} {'lowq':>6s} {'spoof':>6s} | "
          f"{'recall':>8s} {'lowq':>6s} {'spoof':>6s}")
    print("-" * 62)
    for q in qualities:
        r = res[q]
        ts, rg = r["text_splice"], r["regenerate"]
        mark = "  <-- gate" if q == thr else ""
        print(f"{q:7d} | {ts['recall']:8.1%} {ts['lowq_rate']:6.0%} {ts['mean_spoof']:6.3f} | "
              f"{rg['recall']:8.1%} {rg['lowq_rate']:6.0%} {rg['mean_spoof']:6.3f}{mark}")

    lo, hi = min(qualities), max(qualities)
    print(f"\nrecall collapse from Q={lo} to Q={hi}:")
    for a in EVADABLE_ATTACKS:
        print(f"  {a:14s} {res[lo][a]['recall']:6.1%} -> {res[hi][a]['recall']:6.1%}")

    if args.json_out:
        args.json_out.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
