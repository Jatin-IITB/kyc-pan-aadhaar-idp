"""Compression-domain diagnostics for the text_splice / regenerate tautology.

`tools/eval/evasion_probe.py` measures HOW BADLY the metadata gate fails when
the attacker raises their save quality. This tool asks the follow-up question:
is there a pixel/frequency-domain signal that could replace it?

Three modes, in the order they were run for W16 (ADR-040):

  dq      Global double-quantization + blockiness statistics.
          Answers: does textbook double-JPEG detection work here?
          Measured answer: NO, and it is INVERTED -- genuine images are
          mismatched-double-JPEG (augment writes q in [70,95), then
          identity_forge re-saves at 92) while `regenerate` is triple-
          compressed at a SINGLE quality, the near-idempotent case.

  oracle  Local JPEG-ghost contrast using the ground-truth tamper box.
          Answers: does a local signal exist AT ALL for text_splice?
          Being handed the answer, this UPPER-BOUNDS any blind detector.
          Measured answer: YES, strongly, when probing at the host's
          original save quality.

  blind   The same ghost search without the box: sweep probe qualities and
          scan candidate windows, taking the max.
          Answers: does the signal survive multiple-comparison inflation?
          Measured answer: NO with a ratio statistic (it degenerates on
          near-idempotent maps). See ADR-040 for the full result.

    .venv/bin/python -m tools.eval.double_jpeg_probe --mode dq
    .venv/bin/python -m tools.eval.double_jpeg_probe --mode oracle --quality 95
    .venv/bin/python -m tools.eval.double_jpeg_probe --mode blind --stat diff

Requires the synthetic corpus: `make forge` (or run_eval --regen) first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.eval.evasion_probe import forced_jpeg_quality

REPO = Path(__file__).resolve().parents[2]
DOC_TYPES = ("pan", "aadhaar", "driving_license")
SEVERITIES = ("low", "med", "high")
BLOCK = 8
GHOST_PROBE_QS = (70, 78, 85, 88, 92, 95)
# Candidate window sizes in ghost-map cells. ID value fields are wide and short;
# these bracket that shape without encoding any specific box.
WINDOWS = ((3, 16), (3, 28), (4, 20), (4, 36), (6, 24), (6, 44))


# ----------------------------------------------------------------- shared io

def load_cases(split: str):
    cases = []
    for dt in DOC_TYPES:
        tdir = REPO / "data" / split / "synthetic" / dt / "truth"
        idir = REPO / "data" / split / "synthetic" / dt / "images"
        for tp in sorted(tdir.glob("*.json")):
            img = cv2.imread(str(idir / f"{tp.stem}.jpg"))
            if img is not None:
                cases.append((dt, tp.stem, json.loads(tp.read_text()), img))
    if not cases:
        raise SystemExit(
            f"no genuine images under data/{split}/synthetic — run `make forge`")
    return cases


# ------------------------------------------------------------------ mode: dq

def dct_blocks(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    h, w = h - h % 8, w - w % 8
    g = gray[:h, :w].astype(np.float32) - 128.0
    b = g.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return np.stack([cv2.dct(x) for x in b])


def dq_periodicity(coeffs: np.ndarray) -> float:
    """Strongest non-DC spectral peak of a DCT coefficient histogram.

    Mismatched double compression (q1 != q2) leaves a comb pattern in the
    histogram; single or same-q compression does not.
    """
    v = np.round(coeffs).astype(int)
    v = v[np.abs(v) <= 50]
    if v.size < 200:
        return 0.0
    hist = np.bincount(v - v.min(), minlength=101).astype(float)
    if hist.sum() < 1e-6:
        return 0.0
    hist /= hist.sum()
    spec = np.abs(np.fft.rfft(hist - hist.mean()))
    if spec.size < 4:
        return 0.0
    band = spec[2:]
    return float(band.max() / (band.mean() + 1e-9))


def blockiness(gray: np.ndarray) -> float:
    """8x8 grid blocking strength: on-grid vs off-grid gradient energy."""
    g = gray.astype(np.float32)
    dh, dv = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    on = ((dh[:, 7::8].mean() if dh[:, 7::8].size else 0.0)
          + (dv[7::8, :].mean() if dv[7::8, :].size else 0.0)) / 2.0
    mh = np.ones(dh.shape[1], bool); mh[7::8] = False
    mv = np.ones(dv.shape[0], bool); mv[7::8] = False
    off = ((dh[:, mh].mean() if mh.any() else 1e-9)
           + (dv[mv, :].mean() if mv.any() else 1e-9)) / 2.0
    return float(on / (off + 1e-9))


def features(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = dct_blocks(gray)
    dq = np.mean([dq_periodicity(d[:, i, j])
                  for (i, j) in ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))])
    return {"dq": float(dq), "blk": blockiness(gray)}


def run_dq(split: str, quality: int):
    from tools.forge.tamper_forge import attack_regenerate, attack_text_splice

    pops = {"genuine": [], f"regenerate@{quality}": [], f"text_splice@{quality}": []}
    for dt, stem, truth, img in load_cases(split):
        pops["genuine"].append(features(img))
        for sev in SEVERITIES:
            rng = np.random.default_rng(abs(hash((stem, sev))) % 2**32)
            with forced_jpeg_quality(quality):
                r, _l, _d = attack_regenerate(img, truth, rng, sev, dt)
                t, _l, _d = attack_text_splice(img, truth, rng, sev)
            pops[f"regenerate@{quality}"].append(features(r))
            pops[f"text_splice@{quality}"].append(features(t))

    print(f"\n=== global double-quantization / blockiness ({split}) ===")
    print(f"{'population':18s} {'n':>4s} {'dq_mean':>9s} {'dq_p10':>8s} "
          f"{'dq_p90':>8s} {'blk_mean':>9s}")
    for name, rows in pops.items():
        dq = np.array([r["dq"] for r in rows]); bk = np.array([r["blk"] for r in rows])
        print(f"{name:18s} {len(rows):4d} {dq.mean():9.3f} "
              f"{np.percentile(dq,10):8.3f} {np.percentile(dq,90):8.3f} {bk.mean():9.3f}")
    g = np.array([r["dq"] for r in pops["genuine"]]).mean()
    print("\ndelta vs genuine (negative => attack has LESS DQ signal than genuine):")
    for name in pops:
        if name != "genuine":
            print(f"  {name:18s} {np.array([r['dq'] for r in pops[name]]).mean() - g:+.3f}")


# -------------------------------------------------------------- mode: ghost

def ghost_map(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return np.zeros((1, 1), np.float32)
    re = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    resid = np.abs(img.astype(np.float32) - re.astype(np.float32)).mean(axis=2)
    h, w = resid.shape
    h, w = h - h % BLOCK, w - w % BLOCK
    return resid[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(1, 3))


def box_mask(shape, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    r0, r1 = max(0, y1 // BLOCK), min(shape[0], y2 // BLOCK + 1)
    c0, c1 = max(0, x1 // BLOCK), min(shape[1], x2 // BLOCK + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def run_oracle(split: str, quality: int):
    from tools.forge.tamper_forge import attack_text_splice

    contrast = {q: [] for q in GHOST_PROBE_QS}
    control = {q: [] for q in GHOST_PROBE_QS}
    for dt, stem, truth, img in load_cases(split):
        for sev in SEVERITIES:
            rng = np.random.default_rng(abs(hash((stem, sev))) % 2**32)
            with forced_jpeg_quality(quality):
                out, lbl, _d = attack_text_splice(img, truth, rng, sev)
            if lbl.region is None:
                continue
            for q in GHOST_PROBE_QS:
                gm = ghost_map(out, q)
                m = box_mask(gm.shape, lbl.region)
                if m is None or not (~m).any():
                    continue
                contrast[q].append(gm[m].mean() / (gm[~m].mean() + 1e-9))
                gg = ghost_map(img, q)
                mg = box_mask(gg.shape, lbl.region)
                if mg is not None and (~mg).any():
                    control[q].append(gg[mg].mean() / (gg[~mg].mean() + 1e-9))

    print(f"\n=== ORACLE in-box/out-of-box ghost contrast (splice @ Q={quality}) ===")
    print("1.00 = painted region indistinguishable from host\n")
    print(f"{'probe Q':>8s} {'n':>5s} {'SPLICE':>9s} {'p10':>8s} "
          f"{'GENUINE ctrl':>13s} {'delta':>8s}")
    for q in GHOST_PROBE_QS:
        a, c = np.array(contrast[q]), np.array(control[q])
        if not a.size:
            continue
        print(f"{q:8d} {a.size:5d} {a.mean():9.3f} {np.percentile(a,10):8.3f} "
              f"{c.mean():13.3f} {a.mean()-c.mean():+8.3f}")


def max_window_contrast(gm: np.ndarray, stat: str) -> float:
    """Largest window-vs-outside contrast over all positions and sizes.

    stat='ratio' is unusable blind: probing a genuine image at its own save
    quality makes the map near-zero everywhere, the outside mean collapses and
    any noise window yields a huge ratio (measured: all 30 genuine peaked at
    q=92 and outscored the splices). stat='diff' stays in residual units and
    remains ~0 on a near-idempotent map.
    """
    H, W = gm.shape
    total, n = float(gm.sum()), H * W
    ii = np.zeros((H + 1, W + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(gm.astype(np.float64), axis=0), axis=1)
    best = 0.0
    for wh, ww in WINDOWS:
        if wh >= H or ww >= W:
            continue
        s = ii[wh:, ww:] - ii[:-wh, ww:] - ii[wh:, :-ww] + ii[:-wh, :-ww]
        inside = s / (wh * ww)
        outside = (total - s) / max(n - wh * ww, 1)
        v = inside / (outside + 1e-9) if stat == "ratio" else inside - outside
        best = max(best, float(v.max()))
    return best


def run_blind(split: str, quality: int, stat: str):
    from tools.forge.tamper_forge import attack_text_splice

    def score(img):
        return max(max_window_contrast(ghost_map(img, q), stat)
                   for q in GHOST_PROBE_QS if q != 95)

    gen, spl = [], []
    for dt, stem, truth, img in load_cases(split):
        gen.append(score(img))
        for sev in SEVERITIES:
            rng = np.random.default_rng(abs(hash((stem, sev))) % 2**32)
            with forced_jpeg_quality(quality):
                out, _l, _d = attack_text_splice(img, truth, rng, sev)
            spl.append(score(out))

    gen, spl = np.array(gen), np.array(spl)
    print(f"\n=== BLIND sliding-window ghost, stat={stat} (splice @ Q={quality}) ===")
    print(f"{'population':14s} {'n':>4s} {'mean':>9s} {'p10':>9s} {'p90':>9s} {'max':>9s}")
    for name, a in (("genuine", gen), ("text_splice", spl)):
        print(f"{name:14s} {a.size:4d} {a.mean():9.3f} {np.percentile(a,10):9.3f} "
              f"{np.percentile(a,90):9.3f} {a.max():9.3f}")
    zero = [t for t in np.arange(0.05, 60.0, 0.05) if (gen > t).mean() == 0.0]
    if zero:
        t0 = min(zero)
        print(f"\nsmallest 0%-FPR threshold = {t0:.2f} -> recall {(spl > t0).mean():.1%}")
    else:
        print("\nno threshold achieves 0% genuine FPR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dq", "oracle", "blind"), default="dq")
    ap.add_argument("--split", default="holdout", choices=("holdout", "tuning"))
    ap.add_argument("--quality", type=int, default=95,
                    help="attacker save quality (the metadata gate is blind >= 88)")
    ap.add_argument("--stat", choices=("diff", "ratio"), default="diff",
                    help="blind-mode contrast statistic")
    a = ap.parse_args()
    if a.mode == "dq":
        run_dq(a.split, a.quality)
    elif a.mode == "oracle":
        run_oracle(a.split, a.quality)
    else:
        run_blind(a.split, a.quality, a.stat)


if __name__ == "__main__":
    main()
