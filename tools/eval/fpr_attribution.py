"""Attribute genuine-document false positives to individual detectors.

Every forensic threshold in this repo was calibrated on Identity Forge output:
synthetic renders saved at Q=92 with light augmentation, one renderer's
typography, no camera pipeline. Real cards arrive from phone cameras, flatbed
scanners and messaging apps at wildly different quality. When the genuine FPR
jumps on a real corpus, the useful question is not "how much" but "which
detector, and on what feature value".

This walks the genuine set, runs the real detector stack, and reports for each
image which components fired and why -- plus the distribution of the underlying
features, so a threshold can be re-derived rather than guessed at.

    .venv/bin/python -m tools.eval.fpr_attribution --root data/holdout
    .venv/bin/python -m tools.eval.fpr_attribution --root data/holdout --verbose

Reports, per detector: how many genuine documents it fired on, and the feature
distribution driving it. A detector at 0% is safe on this corpus; one near
100% is miscalibrated for it, not necessarily wrong in principle.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def build_stack():
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


def analyse(root: Path, verbose: bool):
    detectors, template_font, scorer = build_stack()

    images = sorted(glob.glob(str(root / "synthetic" / "*" / "images" / "*.jpg")))
    if not images:
        raise SystemExit(f"no genuine images under {root}/synthetic/*/images/")

    fired = Counter()
    flagged = 0
    feat = defaultdict(list)
    rows = []

    for ip in images:
        p = Path(ip)
        doc_type = p.parts[-3]
        img = cv2.imread(ip)
        raw = p.read_bytes()
        if img is None:
            continue

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
        res = scorer.compute(r["ela"], r["copy_move"], r["font"], r["metadata"],
                             r["screen"], font_template_result=ft)

        is_flagged = res["recommendation"] != "PASS"
        flagged += is_flagged
        for comp, score in res["component_scores"].items():
            if score > 0:
                fired[comp] += 1

        # features behind each gate, for re-deriving thresholds
        jq = r["metadata"].get("jpeg_quality")
        if jq is not None:
            feat["jpeg_quality"].append(jq)
        feat["ela_regions"].append(len(r["ela"].get("suspicious_regions") or []))
        feat["moire"].append(r["screen"].get("moire_score", 0.0))
        feat["ring_ratio"].append(r["screen"].get("ring_ratio", 0.0))
        feat["combined"].append(r["screen"].get("combined_score", 0.0))
        feat["cm_detected"].append(1 if r["copy_move"].get("detected") else 0)
        feat["ft_breaches"].append(len(ft.get("breaches") or []))
        feat["spoof"].append(res["spoof_score"])

        rows.append((p.name, doc_type, is_flagged, res["spoof_score"],
                     [e["type"] for e in res["evidence"]], jq))

    n = len(rows)
    print(f"\n=== genuine FPR attribution — {root} ===")
    print(f"{n} genuine documents, {flagged} flagged "
          f"({flagged / max(n, 1):.1%} FPR)\n")

    print(f"{'detector':16s} {'fired':>6s} {'rate':>8s}")
    for comp in ("metadata", "screen", "ela", "copy_move", "font_template", "font"):
        c = fired.get(comp, 0)
        print(f"  {comp:14s} {c:6d} {c / max(n, 1):8.1%}")

    def dist(key, fmt="{:.3f}"):
        v = np.array(feat[key], dtype=float)
        if not v.size:
            return "n/a"
        return (f"min {fmt.format(v.min())}  p10 {fmt.format(np.percentile(v, 10))}  "
                f"p50 {fmt.format(np.percentile(v, 50))}  "
                f"p90 {fmt.format(np.percentile(v, 90))}  max {fmt.format(v.max())}")

    print("\nfeature distributions on GENUINE documents:")
    print(f"  jpeg_quality   {dist('jpeg_quality', '{:.0f}')}")
    print("                 (gate fires below 88 — synthetic forge writes 92)")
    print(f"  moire_score    {dist('moire')}")
    print(f"  ring_ratio     {dist('ring_ratio')}")
    print(f"  combined       {dist('combined')}")
    print(f"  ela_regions    {dist('ela_regions', '{:.0f}')}")
    print(f"  ft_breaches    {dist('ft_breaches', '{:.0f}')}")
    print(f"  spoof_score    {dist('spoof')}")
    print(f"  copy_move fired on {int(np.sum(feat['cm_detected']))}/{n}")

    if feat["jpeg_quality"]:
        q = np.array(feat["jpeg_quality"])
        below = int((q < 88).sum())
        print(f"\njpeg_quality < 88 on {below}/{len(q)} genuine documents "
              f"({below / len(q):.1%}).")
        if below:
            print("  This alone sets spoof_score to >= 0.27 via the metadata prior,")
            print("  which is above the 0.2 PASS threshold. ADR-028's tautology")
            print("  cuts both ways: it manufactures recall on forge output and")
            print("  manufactures false positives on real scans.")

    if verbose:
        print(f"\n{'file':38s} {'type':16s} {'q':>4s} {'spoof':>7s}  evidence")
        for name, dt, fl, s, ev, jq in rows:
            if fl:
                print(f"  {name:36s} {dt:16s} {str(jq or '-'):>4s} {s:7.3f}  "
                      f"{','.join(ev)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/holdout",
                    help="split root containing synthetic/<doc_type>/images/")
    ap.add_argument("--verbose", action="store_true",
                    help="list every flagged genuine document with its evidence")
    a = ap.parse_args()
    analyse(Path(a.root), a.verbose)


if __name__ == "__main__":
    main()
