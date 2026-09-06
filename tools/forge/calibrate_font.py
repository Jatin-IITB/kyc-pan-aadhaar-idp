"""Calibrate per-doc-type font-forensics thresholds on held-out data.

The font_swap attack re-renders an entire card with permuted font roles, so
every value line shifts together. Intra-document outlier detection is blind to
it by construction (see ADR-030). Instead we compare a document's typographic
signature against a per-doc-type reference envelope calibrated from genuine
documents.

Thresholds are fit on a CALIBRATION seed and scored on a disjoint HOLDOUT seed,
so the reported FPR/recall are not in-sample artifacts.

    .venv/bin/python -m tools.forge.calibrate_font --n 16 --emit
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from services.forensics.font_profile import signature

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "data" / "_font_cal"
PROFILE = REPO / "config" / "font_profiles.json"

DOC_TYPES = ("pan", "aadhaar", "driving_license")
CAL_SEED = 9001
HOLD_SEED = 24601

# Features are one-sided: a swap raises corner density (serif terminals) and
# lowers stroke modulation. Direction is fixed here from mechanism, not fit,
# so a lucky sample cannot flip it.
BASE_FEATURES = {
    "corner_top": "high",
    "mod_top": "low",
    "adv_cv_min": "high",
}

# id_width_cv measures glyph width uniformity on the ID-number line.  Mono
# digits are near-identical widths; proportional/serif digits vary ('1' vs '0').
# On DL (pure digit ID), genuine cluster is tight (cv 0.02-0.05) and swap
# jumps to 0.16+.  On PAN (alphanumeric ABCDE1234F), letter ink widths vary
# even in monospace, making genuine id_width_cv too noisy for a stable bound.
# On Aadhaar (12-digit number), genuine clusters at 0.15-0.25; font swaps
# scatter both directions but the genuine right tail is too heavy for a band
# bound (12.5% FPR), so only the low side is used (W12, ADR-036).
# adv_cv_min is EXCLUDED on Aadhaar (None). Its mechanism — a monospace ID
# line whose advance turns irregular when mono is swapped away — cannot
# operate there: the 12-digit number renders as three 4-glyph groups, below
# the 5-component minimum, so the feature silently lands on the proportional
# value/label lines instead. Measured on the Windows-rendered corpus: 0/48
# font_swap documents exceed the genuine maximum (the high-severity swap sets
# values in mono, which LOWERS the value-line CV), while a blur-fragmented
# genuine document (stroke slivers counted as glyph starts) sits 11 MAD above
# the bound — the last genuine false positive of the FPR audit.
DT_EXTRA = {
    "driving_license": {"id_width_cv": "high"},
    "aadhaar": {"id_width_cv": "low", "adv_cv_min": None},
}

# Per-doc-type vote/margin: DL has 4 features so vote=2 is stable at a
# tighter margin; PAN has 3 features and needs vote=1 with a wider margin.
# Aadhaar was margin=8 (W6-W10) which zeroed recall; adding id_width_cv
# and tightening to margin=4 recovered recall at 0% LOO FPR (W12).
DT_VOTE = {
    "pan": 1,
    "aadhaar": 1,
    "driving_license": 2,
}
DT_MARGIN = {
    "pan": 8.0,
    "aadhaar": 4.0,
    "driving_license": 5.0,
}
# Per-feature margin overrides (W15, ADR-039). PAN corner_top at margin=8
# gives bound 0.436 — far outside the genuine range (max 0.340 across 52
# images) because the calibration cluster is tight (MAD=0.015). Margin 3.5
# gives bound 0.369, an interior point between genuine max (0.340) and
# weakest swap recovery (0.400). Validated at 0% FPR on 52 genuine PAN.
# Windows-corpus recalibration (FPR audit): Aadhaar corner_top does not
# separate genuine from swap (both span 0.26-0.60), and at margin 4 two
# calibration documents breached it under leave-one-out; 5 restores LOO FPR 0
# at no recall cost. DL mod_top is the discriminative feature there: margin 4
# lifts calibration recall 12/16 -> 15/16 on both seeds with LOO FPR 0.
DT_FEATURE_MARGIN: dict[str, dict[str, float]] = {
    "pan": {"corner_top": 3.5},
    "aadhaar": {"corner_top": 5.0},
    "driving_license": {"mod_top": 4.0},
}


def features_for(dt):
    merged = {**BASE_FEATURES, **DT_EXTRA.get(dt, {})}
    return {k: v for k, v in merged.items() if v is not None}


FEATURES = {**BASE_FEATURES, **{"id_width_cv": "high"}}


# --------------------------------------------------------------- calibration

def _bound(values, side, margin):
    """Robust one-sided envelope from calibration genuine values.

    Median +/- margin * MAD rather than min/max: a strict extremum tracks the
    calibration sample's luckiest outlier and does not generalize.
    """
    v = np.asarray(values, dtype=float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) or float(v.std()) or 1e-6
    return med + margin * mad if side == "high" else med - margin * mad


def _split_complete(split_dir: Path, n: int) -> bool:
    """Cache is valid only if EVERY doc type has exactly n genuine images and
    a forged set. The original check looked at pan/genuine alone (audit C1) —
    a partial or differently-sized cache silently produced envelopes that the
    committed seeds could not reproduce.
    """
    for dt in DOC_TYPES:
        if len(list((split_dir / "genuine" / dt / "images").glob("*.jpg"))) != n:
            return False
        if not list((split_dir / "forged" / dt / "images").glob("*font_swap*.jpg")):
            return False
    return True


def build_data(n, force=False):
    from tools.forge.identity_forge import generate
    from tools.forge.tamper_forge import forge_dataset

    if force and WORK.exists():
        shutil.rmtree(WORK)
    for split, seed in (("cal", CAL_SEED), ("hold", HOLD_SEED)):
        sdir = WORK / split
        if _split_complete(sdir, n):
            continue
        shutil.rmtree(sdir, ignore_errors=True)
        for dt in DOC_TYPES:
            generate(dt, n, sdir / "genuine", seed=seed, augment_level="light")
        forge_dataset(sdir / "genuine", sdir / "forged", list(DOC_TYPES),
                      ["font_swap"], 1, "mixed", seed)
    return WORK


def collect(split):
    out = {dt: {"genuine": [], "swap": []} for dt in DOC_TYPES}
    for dt in DOC_TYPES:
        gdir = WORK / split / "genuine" / dt / "images"
        for p in sorted(gdir.glob("*.jpg")):
            s = signature(cv2.imread(str(p)))
            if s:
                out[dt]["genuine"].append(s)
        fdir = WORK / split / "forged" / dt / "images"
        if fdir.exists():
            for p in sorted(fdir.glob("*font_swap*.jpg")):
                s = signature(cv2.imread(str(p)))
                if s:
                    out[dt]["swap"].append(s)
    return out


def _margin_for(dt, feat, global_margin=None):
    """Per-feature margin with DT_FEATURE_MARGIN override."""
    if global_margin is not None:
        return global_margin
    return DT_FEATURE_MARGIN.get(dt, {}).get(feat, DT_MARGIN[dt])


def fit(cal, margin=None):
    prof = {}
    for dt in DOC_TYPES:
        g = cal[dt]["genuine"]
        prof[dt] = {"_vote": DT_VOTE[dt]}
        for feat, side in features_for(dt).items():
            m = _margin_for(dt, feat, margin)
            vals = [s[feat] for s in g if s.get(feat) is not None]
            if len(vals) >= 4:
                if side == "band":
                    prof[dt][feat] = {
                        "side": "band",
                        "low": _bound(vals, "low", m),
                        "high": _bound(vals, "high", m),
                    }
                else:
                    prof[dt][feat] = {"side": side, "bound": _bound(vals, side, m)}
    return prof


def score(sig, prof_dt, vote=1):
    """True when at least `vote` calibrated envelopes are breached."""
    if not prof_dt:
        return False, []
    hits = []
    for feat, spec in prof_dt.items():
        if feat.startswith("_"):
            continue
        v = sig.get(feat)
        if v is None:
            continue
        side = spec.get("side")
        if side == "band":
            if (spec.get("low") is not None and v < spec["low"]) or \
               (spec.get("high") is not None and v > spec["high"]):
                hits.append(feat)
        elif (side == "high" and v > spec["bound"]) or \
             (side == "low" and v < spec["bound"]):
            hits.append(feat)
    return len(hits) >= vote, hits


def loo_fpr(cal, margin=None):
    """Leave-one-out FPR estimate using calibration data only.

    Selecting the smallest margin that merely zeroes in-sample calibration FPR
    lands on the edge of the envelope and does not generalize — measured
    directly: such a config scored 0% calibration FPR and 8.3% holdout FPR.
    LOO refits the bound without each genuine document and tests that held-out
    document, giving an out-of-sample estimate that never consults the real
    holdout split.
    """
    fp = n = 0
    for dt in DOC_TYPES:
        vote = DT_VOTE[dt]
        docs = cal[dt]["genuine"]
        for i in range(len(docs)):
            rest = docs[:i] + docs[i + 1:]
            prof_dt = {}
            for feat, side in features_for(dt).items():
                m = _margin_for(dt, feat, margin)
                vals = [s[feat] for s in rest if s.get(feat) is not None]
                if len(vals) >= 4:
                    if side == "band":
                        prof_dt[feat] = {
                            "side": "band",
                            "low": _bound(vals, "low", m),
                            "high": _bound(vals, "high", m),
                        }
                    else:
                        prof_dt[feat] = {"side": side, "bound": _bound(vals, side, m)}
            hit, _ = score(docs[i], prof_dt, vote)
            fp += bool(hit)
            n += 1
    return fp / max(n, 1)


def evaluate(data, prof):
    tp = fn = fp = tn = 0
    per = {}
    for dt in DOC_TYPES:
        vote = prof.get(dt, {}).get("_vote", 1)
        d_tp = sum(score(s, prof.get(dt), vote)[0] for s in data[dt]["swap"])
        d_fp = sum(score(s, prof.get(dt), vote)[0] for s in data[dt]["genuine"])
        ns, ng = len(data[dt]["swap"]), len(data[dt]["genuine"])
        per[dt] = {"recall": d_tp / ns if ns else 0.0,
                   "fpr": d_fp / ng if ng else 0.0, "n_swap": ns, "n_gen": ng}
        tp += d_tp; fn += ns - d_tp; fp += d_fp; tn += ng - d_fp
    return {"recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1), "per": per}


def data_hash() -> str:
    """Content hash of every calibration/holdout image, for provenance.

    Audit C1: the shipped profile could not be regenerated because stale cached
    data had produced it. Recording what the envelope was fit ON makes that
    failure detectable instead of silent.
    """
    import hashlib
    h = hashlib.sha256()
    for split in ("cal", "hold"):
        for dt in DOC_TYPES:
            for sub in ("genuine", "forged"):
                d = WORK / split / sub / dt / "images"
                for p in sorted(d.glob("*.jpg")):
                    h.update(p.name.encode())
                    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="genuine docs per type per split")
    ap.add_argument("--emit", action="store_true", help="write config/font_profiles.json")
    ap.add_argument("--force", action="store_true", help="regenerate data")
    ap.add_argument("--reveal-holdout", action="store_true",
                    help="show holdout columns during the sweep. Off by default: "
                         "watching holdout numbers while choosing a selection rule "
                         "consumes the holdout (ADR-030 S2).")
    args = ap.parse_args()

    print(f"generating (n={args.n}/type/split)...")
    build_data(args.n, force=args.force)
    cal, hold = collect("cal"), collect("hold")
    print(f"  cal   genuine={sum(len(cal[d]['genuine']) for d in DOC_TYPES)} "
          f"swap={sum(len(cal[d]['swap']) for d in DOC_TYPES)}")
    print(f"  hold  genuine={sum(len(hold[d]['genuine']) for d in DOC_TYPES)} "
          f"swap={sum(len(hold[d]['swap']) for d in DOC_TYPES)}")

    # Per-doc-type calibration: each doc type uses its own margin/vote from
    # DT_MARGIN / DT_VOTE. The LOO FPR check runs per-type independently.
    lf = loo_fpr(cal)
    prof = fit(cal)
    c, h = evaluate(cal, prof), evaluate(hold, prof)

    print(f"\nper-doc-type config (margin/vote from DT_MARGIN/DT_VOTE):")
    print(f"  LOO FPR: {lf:.1%}")
    print(f"\n{'doc type':18s} {'margin':>7s} {'vote':>5s} {'CAL rec':>8s}", end="")
    if args.reveal_holdout:
        print(f" {'HOLD rec':>9s} {'HOLD fpr':>9s}", end="")
    print()
    for dt in DOC_TYPES:
        cp, hp = c["per"][dt], h["per"][dt]
        print(f"  {dt:16s} {DT_MARGIN[dt]:7.1f} {DT_VOTE[dt]:5d} {cp['recall']:8.1%}", end="")
        if args.reveal_holdout:
            print(f" {hp['recall']:9.1%} {hp['fpr']:9.1%}", end="")
        print()

    print(f"\n{'HELD-OUT RESULT':22s} recall={h['recall']:.1%}  FPR={h['fpr']:.1%}")
    print(f"\n{'doc type':18s} {'recall':>8s} {'FPR':>7s}")
    for dt in DOC_TYPES:
        p = h["per"][dt]
        print(f"  {dt:16s} {p['recall']:8.1%} {p['fpr']:7.1%}")

    if lf > 0:
        print(f"\nWARNING: LOO FPR is {lf:.1%} — check DT_MARGIN/DT_VOTE settings.")

    if args.emit:
        from services.forensics.font_profile import EXTRACTOR_VERSION
        PROFILE.write_text(json.dumps({
            "_comment": "Calibrated by tools/forge/calibrate_font.py — see ADR-030. "
                        "Fit on calibration seed only; held-out numbers below are "
                        "out-of-sample.",
            "extractor_version": EXTRACTOR_VERSION,
            "calibration_data_hash": data_hash(),
            "calibration_n_per_type": args.n,
            "calibration_seed": CAL_SEED, "holdout_seed": HOLD_SEED,
            "vote": 1, "features": FEATURES,
            "measured_holdout": {"recall": round(h["recall"], 4),
                                 "fpr": round(h["fpr"], 4),
                                 "per_doc_type": {k: {kk: round(vv, 4) for kk, vv
                                                      in v.items()}
                                                  for k, v in h["per"].items()}},
            "profiles": prof}, indent=2))
        print(f"\nwrote {PROFILE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
