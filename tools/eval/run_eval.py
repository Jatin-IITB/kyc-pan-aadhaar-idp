"""Truth Engine eval harness (Phase 11 W3).

    make eval        # full run: forensics + decision + VLM extraction + gates
    make eval-fast   # skip the VLM tier (no Ollama needed)

Ensures the synthetic + tamper datasets exist (deterministic seeds), sweeps the
forensic suite over genuine and forged documents, pushes every document through
the REAL decision layer (calibrator + auto-clear overrides) to measure spoof
leakage, optionally runs VLM extraction against ground truth, then writes
eval/metrics.json + eval/report.html and checks config/eval_thresholds.yaml.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml

from tools.eval.metrics import check_gates, score_extraction

GENUINE_SEED, TAMPER_SEED, PER_TYPE = 42, 123, 10


def ensure_datasets(root: Path, regen: bool) -> None:
    from tools.forge.identity_forge import generate
    from tools.forge.tamper_forge import ATTACKS, forge_dataset
    from tools.forge.templates import RENDERERS

    syn, tam = root / "synthetic", root / "tamper"
    for doc_type in RENDERERS:
        if regen or not (syn / doc_type / "manifest.jsonl").exists():
            generate(doc_type, PER_TYPE, syn, seed=GENUINE_SEED, augment_level="light")
    if regen or not (tam / "manifest.jsonl").exists():
        forge_dataset(syn, tam, list(RENDERERS), ATTACKS, per_doc=1,
                      severity="mixed", seed=TAMPER_SEED)


def _pctl(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(values, n=100)[int(q) - 1]) if len(values) > 1 else values[0]


def run_forensics(root: Path) -> Dict[str, Any]:
    from services.forensics.copy_move import CopyMoveDetector
    from services.forensics.ela import ELADetector
    from services.forensics.font_analysis import FontConsistencyAnalyzer
    from services.forensics.metadata import MetadataForensics
    from services.forensics.screen_recapture import ScreenRecaptureDetector
    from services.forensics.spoof_scorer import SpoofScorer

    detectors = {
        "ela": ELADetector(), "copy_move": CopyMoveDetector(),
        "font": FontConsistencyAnalyzer(), "metadata": MetadataForensics(),
        "screen": ScreenRecaptureDetector(),
    }
    scorer = SpoofScorer()
    latency: Dict[str, List[float]] = defaultdict(list)

    def sweep(image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        raw = Path(image_path).read_bytes()
        results = {}
        for name, det in detectors.items():
            t0 = time.perf_counter()
            if name == "metadata":
                results[name] = det.analyze(raw)
            elif name in ("ela", "font"):
                results[name] = det.analyze(img) if name == "ela" else det.analyze(img, [])
            else:
                results[name] = det.detect(img)
            latency[name].append((time.perf_counter() - t0) * 1000)
        return scorer.compute(results["ela"], results["copy_move"], results["font"],
                              results["metadata"], results["screen"])

    genuine: List[Dict[str, Any]] = []
    for ip in sorted(glob.glob(str(root / "synthetic" / "*" / "images" / "*.jpg"))):
        genuine.append(sweep(ip))

    per_attack: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ap in sorted(glob.glob(str(root / "tamper" / "*" / "attacks" / "*.json"))):
        rec = json.loads(Path(ap).read_text())
        ip = ap.replace("/attacks/", "/images/").replace(".json", ".jpg")
        per_attack[rec["attack"]].append(sweep(ip))

    if not genuine or not per_attack:
        raise SystemExit(
            f"eval dataset empty (genuine={len(genuine)}, "
            f"attack_classes={len(per_attack)}) — regenerate with `make forge`")

    genuine_fp = sum(1 for r in genuine if r["recommendation"] != "PASS")
    attack_stats = {}
    hits = total = 0
    for attack, results in sorted(per_attack.items()):
        h = sum(1 for r in results if r["recommendation"] != "PASS")
        attack_stats[attack] = {"recall": round(h / len(results), 4),
                                "flagged": h, "n": len(results)}
        hits += h
        total += len(results)

    return {
        "genuine_n": len(genuine),
        "genuine_fpr": round(genuine_fp / len(genuine), 4) if genuine else None,
        "overall_recall": round(hits / total, 4) if total else None,
        "per_attack": attack_stats,
        "latency_ms": {k: {"p50": round(_pctl(v, 50), 1), "p95": round(_pctl(v, 95), 1)}
                       for k, v in latency.items()},
        "_spoof_scores": {
            "genuine": [r["spoof_score"] for r in genuine],
            "tampered": {a: [r["spoof_score"] for r in rs] for a, rs in per_attack.items()},
        },
    }


def run_decision(forensics: Dict[str, Any]) -> Dict[str, Any]:
    """Feed measured spoof scores through the REAL decision layer.

    Extraction/policy/cross-doc are held at 1.0 (perfect), isolating the
    question: can a forged document slip past calibration + overrides?
    """
    from services.decisioning.auto_clear import AutoClearEngine
    from services.decisioning.calibrator import ConfidenceCalibrator

    calibrator, engine = ConfidenceCalibrator(), AutoClearEngine()

    def decide(spoof: float) -> Dict[str, Any]:
        cal = calibrator.calibrate(extraction_score=1.0, forensics_score=spoof)
        return engine.evaluate(cal, spoof_score=spoof)

    scores = forensics.pop("_spoof_scores")
    genuine_dec = [decide(s) for s in scores["genuine"]]
    genuine_ac = sum(1 for d in genuine_dec if d["auto_cleared"])

    # Two distinct failure modes (must not be conflated):
    #   flagged_leakage      — forensics flagged it (spoof >= 0.2) yet it still
    #                          auto-cleared: a DECISION-LAYER bug. Gate: 0, always.
    #   undetected_autoclear — forensics saw nothing (blind spot), so it cleared:
    #                          the end-to-end cost of the W4 recall gap. Ratchet
    #                          gate at the current measured ceiling.
    flagged_leak = 0
    undetected = 0
    tampered_n = 0
    undetected_by_attack: Dict[str, int] = {}
    for attack, spoofs in scores["tampered"].items():
        for s in spoofs:
            tampered_n += 1
            if not decide(s)["auto_cleared"]:
                continue
            if s >= 0.2:
                flagged_leak += 1
            else:
                undetected += 1
                undetected_by_attack[attack] = undetected_by_attack.get(attack, 0) + 1

    return {
        "assumption": "extraction/policy/cross_doc held at 1.0 (isolates forensic gate)",
        "genuine_auto_clear_rate": round(genuine_ac / len(genuine_dec), 4) if genuine_dec else None,
        "flagged_leakage": flagged_leak,
        "undetected_autoclear": undetected,
        "undetected_by_attack": undetected_by_attack,
        "tampered_n": tampered_n,
    }


def run_extraction(root: Path, n: int) -> Optional[Dict[str, Any]]:
    from services.extraction.vlm_extractor import VLMConfig, VLMExtractor

    cfg = yaml.safe_load(Path("config/models.yaml").read_text()) or {}
    vlm_cfg = cfg.get("vlm", {})
    vlm = VLMExtractor(config=VLMConfig(model=vlm_cfg.get("model", "minicpm-v"),
                                        timeout_s=float(vlm_cfg.get("timeout_s", 90))))

    # Warm the model first: a cold Ollama load can exceed the per-call timeout.
    import numpy as np
    warm = VLMExtractor(config=VLMConfig(model=vlm.config.model, timeout_s=300))
    try:
        print("  warming VLM...", flush=True)
        warm.extract_fields(np.full((64, 64, 3), 255, np.uint8), "pan")
    except Exception as e:
        print(f"  extraction tier unavailable ({e}) — skipping", file=sys.stderr)
        return None

    # Balanced round-robin across doc types so no single type dominates the F1
    # (audit MINOR: sorted-stride skewed PAN to 2/12).
    by_type: Dict[str, List[str]] = defaultdict(list)
    for tp in sorted(glob.glob(str(root / "synthetic" / "*" / "truth" / "*.json"))):
        by_type[Path(tp).parent.parent.name].append(tp)
    sampled: List[str] = []
    idx = 0
    while len(sampled) < n and any(idx < len(v) for v in by_type.values()):
        for t in sorted(by_type):
            if idx < len(by_type[t]) and len(sampled) < n:
                sampled.append(by_type[t][idx])
        idx += 1

    samples, lat, failed = [], [], []
    for tp in sampled:
        truth = json.loads(Path(tp).read_text())
        ip = tp.replace("/truth/", "/images/").replace(".json", ".jpg")
        img = cv2.imread(ip)
        t0 = time.perf_counter()
        try:
            ext = vlm.extract_fields(img, truth["doc_type"])
        except Exception as e:
            failed.append(truth["fields"])
            print(f"  {truth['sample_id']} failed ({e}) — counts as all-FN", file=sys.stderr)
            continue
        lat.append(time.perf_counter() - t0)
        samples.append({
            "truth": truth["fields"],
            "predicted": {k: v.get("value", "") for k, v in ext.items()},
        })
        print(f"  extracted {truth['sample_id']} ({lat[-1]:.1f}s)")

    if not samples:
        print("  no extractions succeeded — tier skipped", file=sys.stderr)
        return None

    # Honest scoring (C3): a timeout/failure is not a free pass — each failed
    # doc counts as all-fields-missed (empty predictions => FN). This is the
    # number the gate checks; F1-on-succeeded is reported alongside for context.
    honest = score_extraction(samples + [{"truth": t, "predicted": {}} for t in failed])
    succeeded = score_extraction(samples)
    honest["micro_succeeded"] = succeeded["micro"]
    honest["micro_fuzzy_succeeded"] = succeeded["micro_fuzzy"]
    honest["n_failed"] = len(failed)
    honest["n_attempted"] = len(sampled)
    honest["latency_s"] = {"p50": round(_pctl(lat, 50), 2), "p95": round(_pctl(lat, 95), 2)}
    return honest


def render_report(metrics: Dict[str, Any], out: Path) -> None:
    f, d, e, g = (metrics.get("forensics"), metrics.get("decision"),
                  metrics.get("extraction"), metrics.get("gates"))

    def card(label, value, good=True):
        color = "#10b981" if good else "#ef4444"
        return (f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value" style="color:{color}">{value}</div></div>')

    def pct(x):
        return f'{x*100:.0f}%' if x is not None else "n/a"

    cards = ""
    if f:
        cards += card("Genuine FPR", pct(f["genuine_fpr"]), (f["genuine_fpr"] or 0) == 0)
        cards += card("Tamper Recall", pct(f["overall_recall"]), (f["overall_recall"] or 0) >= 0.4)
    if d:
        cards += card("Flagged Leakage", d["flagged_leakage"], d["flagged_leakage"] == 0)
        cards += card("Genuine Auto-Clear", pct(d.get("genuine_auto_clear_rate")),
                      (d.get("genuine_auto_clear_rate") or 0) >= 0.5)
        cards += card("Blind-Spot Clears", d["undetected_autoclear"], d["undetected_autoclear"] <= 45)
    if e:
        cards += card("Field F1 (honest)", pct(e["micro"]["f1"]), e["micro"]["f1"] >= 0.85)
        if e.get("n_failed"):
            cards += card("Extract Failures", f'{e["n_failed"]}/{e["n_attempted"]}', e["n_failed"] == 0)

    attack_rows = "".join(
        f'<tr><td>{a}</td><td>{s["recall"]*100:.0f}%</td><td>{s["flagged"]}/{s["n"]}</td></tr>'
        for a, s in (f["per_attack"].items() if f else [])
    )
    field_rows = "".join(
        f'<tr><td>{fl}</td><td>{s["f1"]:.3f}</td><td>{s["fuzzy_f1"]:.3f}</td><td>{s["n"]}</td></tr>'
        for fl, s in (e["per_field"].items() if e else [])
    ) or '<tr><td colspan="4">extraction tier not run</td></tr>'
    gate_rows = "".join(
        f'<tr><td>{r["gate"]}</td><td>{r["limit"]}</td><td>{r["actual"]}</td>'
        f'<td class="{r["status"].lower()}">{r["status"]}</td></tr>'
        for r in (g["results"] if g else [])
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Truth Engine Report</title>
<style>
body{{background:#0a0a0b;color:#d4d4d8;font:14px/1.5 -apple-system,sans-serif;max-width:860px;margin:40px auto;padding:0 20px}}
h1{{color:#fff;letter-spacing:-.02em}} h2{{color:#a1a1aa;font-size:13px;text-transform:uppercase;letter-spacing:.08em;margin-top:36px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap}} .card{{background:#18181b;border:1px solid #27272a;border-radius:12px;padding:16px 22px;min-width:130px}}
.label{{font-size:11px;color:#71717a;text-transform:uppercase;letter-spacing:.06em}} .value{{font-size:26px;font-weight:700;margin-top:4px}}
table{{border-collapse:collapse;width:100%;margin-top:10px}} td,th{{border-bottom:1px solid #27272a;padding:8px 10px;text-align:left;font-size:13px}}
th{{color:#71717a;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
.pass{{color:#10b981;font-weight:600}} .fail{{color:#ef4444;font-weight:600}} .skipped{{color:#71717a}}
.meta{{color:#52525b;font-size:12px;margin-top:30px}}
</style></head><body>
<h1>Truth Engine — Eval Report</h1>
<div class="cards">{cards}</div>
<h2>Tamper recall by attack</h2><table><tr><th>Attack</th><th>Recall</th><th>Flagged</th></tr>{attack_rows}</table>
<h2>Extraction by field</h2><table><tr><th>Field</th><th>F1 (exact)</th><th>F1 (fuzzy)</th><th>n</th></tr>{field_rows}</table>
<h2>CI gates</h2><table><tr><th>Gate</th><th>Limit</th><th>Actual</th><th>Status</th></tr>{gate_rows}</table>
<p class="meta">dataset: {metrics["dataset"]} &middot; decision assumption: {d["assumption"] if d else "-"}</p>
</body></html>"""
    (out / "report.html").write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser(description="Truth Engine eval harness")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("eval"))
    ap.add_argument("--regen", action="store_true", help="force dataset regeneration")
    ap.add_argument("--no-extraction", action="store_true", help="skip the VLM tier")
    ap.add_argument("--extraction-n", type=int, default=12)
    ap.add_argument("--check", action="store_true", help="exit non-zero if any gate fails")
    args = ap.parse_args()

    print("[1/5] ensuring datasets...")
    ensure_datasets(args.data_root, args.regen)

    print("[2/5] forensic sweep (genuine + tampered)...")
    forensics = run_forensics(args.data_root)

    print("[3/5] decision layer (calibrator + overrides)...")
    decision = run_decision(forensics)

    extraction = None
    if not args.no_extraction:
        print(f"[4/5] VLM extraction tier (n={args.extraction_n})...")
        extraction = run_extraction(args.data_root, args.extraction_n)
    else:
        print("[4/5] VLM extraction tier skipped")

    metrics: Dict[str, Any] = {
        "dataset": f"seeds genuine={GENUINE_SEED} tamper={TAMPER_SEED}, {PER_TYPE}/type",
        "forensics": forensics,
        "decision": decision,
        "extraction": extraction,
    }
    thresholds = yaml.safe_load(Path("config/eval_thresholds.yaml").read_text()) or {}
    metrics["gates"] = check_gates(metrics, thresholds)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    render_report(metrics, args.out)

    print(f"[5/5] wrote {args.out}/metrics.json and {args.out}/report.html")
    print()
    for r in metrics["gates"]["results"]:
        print(f'  {r["status"]:<8} {r["gate"]:<32} limit={r["limit"]} actual={r["actual"]}')
    print()
    ok = metrics["gates"]["passed"]
    print("ALL GATES PASS" if ok else "GATE FAILURES — see eval/report.html")
    return 0 if ok or not args.check else 1


if __name__ == "__main__":
    sys.exit(main())
