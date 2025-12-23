#!/usr/bin/env python3
# tools/eval_harness/run_eval.py
import argparse
import base64
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# --- Config dataclass ---
@dataclass(frozen=True)
class EvalConfig:
    gateway_url: str
    endpoint: str
    test_dir: Path
    name: str
    timeout_s: int
    max_images: Optional[int]
    allow_base64_fallback: bool
    batch_size: int
    batch_delay_s: float
    doc_type: Optional[str]


# --- Helpers ---
def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _pip_freeze() -> Optional[List[str]]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL)
        lines = [ln.strip() for ln in out.decode("utf-8").splitlines() if ln.strip()]
        return lines
    except Exception:
        return None


def _encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _iter_images(test_dir: Path) -> List[Path]:
    imgs = [p for p in sorted(test_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return imgs


def _chunk(xs: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        return [xs]
    return [xs[i : i + batch_size] for i in range(0, len(xs), batch_size)]


# --- HTTP posting helpers ---
def _post_extract_batch_multipart(
    images: List[Path], url: str, timeout_s: int, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    files = []
    opened = []
    try:
        for p in images:
            f = open(p, "rb")
            opened.append(f)
            files.append(("files", (p.name, f, "application/octet-stream")))
        r = requests.post(url, files=files, params=params, timeout=timeout_s)
        r.raise_for_status()
        return r.json()
    finally:
        for f in opened:
            try:
                f.close()
            except Exception:
                pass


def _post_extract_batch_base64(
    images: List[Path], url: str, timeout_s: int, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    payload = {"images_b64": [_encode_image_b64(p) for p in images]}
    r = requests.post(url, json=payload, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


# --- Result parsing helpers ---
def _get_failure_reason(item: Dict[str, Any]) -> str:
    if not item.get("ok", False):
        return f"transport:{item.get('error', 'unknown')}"
    res = item.get("result") or {}
    val = res.get("validation") or {}
    # accept both naming conventions defensively
    if bool(val.get("is_valid", val.get("isvalid", False))):
        return "valid"
    msg = (val.get("message") or "").strip()
    if not msg:
        return "invalid:unknown"
    lower = msg.lower()
    if "does not match" in lower:
        return "invalid:regex_mismatch"
    if "is not one of" in lower:
        return "invalid:enum"
    if "missing" in lower:
        return "invalid:missing"
    return "invalid:other"


def _extract_doctype(item: Dict[str, Any]) -> str:
    if not item.get("ok", False):
        return "unknown"
    res = item.get("result") or {}
    # accept either document_type or documenttype (defensive)
    dt = res.get("document_type", res.get("documenttype"))
    return dt if isinstance(dt, str) and dt else "unknown"


def _flatten_field_failures(item: Dict[str, Any]) -> List[str]:
    if not item.get("ok", False):
        return []
    res = item.get("result") or {}
    extraction = res.get("extraction") or {}
    failures = []
    if isinstance(extraction, dict):
        for fname, fobj in extraction.items():
            if isinstance(fobj, dict) and ("valid" in fobj) and (not bool(fobj.get("valid"))):
                failures.append(str(fname))
    return failures


def _compute_metrics(response: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    items = response.get("results", [])
    if not isinstance(items, list):
        items = []

    total = len(items)
    ok_count = 0
    valid_count = 0

    by_doctype: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    field_fail_counts: Dict[str, int] = {}

    rows: List[Dict[str, Any]] = []

    for it in items:
        if not isinstance(it, dict):
            continue

        fname = it.get("filename")
        ok = bool(it.get("ok", False))
        ok_count += int(ok)

        dt = _extract_doctype(it)
        by_doctype[dt] = by_doctype.get(dt, 0) + 1

        reason = _get_failure_reason(it)
        by_reason[reason] = by_reason.get(reason, 0) + 1

        res = it.get("result") or {}
        val = res.get("validation") or {}
        is_valid = bool(val.get("is_valid", val.get("isvalid", False))) if ok else False
        valid_count += int(is_valid)

        msg = ""
        if ok:
            msg = (val.get("message") or "") if isinstance(val, dict) else ""
        err = it.get("error") if not ok else ""

        for f in _flatten_field_failures(it):
            field_fail_counts[f] = field_fail_counts.get(f, 0) + 1

        rows.append(
            {
                "filename": fname,
                "ok": ok,
                "doctype": dt,
                "is_valid": is_valid,
                "failure_reason": reason,
                "message": msg,
                "error": err,
            }
        )

    metrics = {
        "total": total,
        "ok_count": ok_count,
        "ok_rate": (ok_count / total) if total else 0.0,
        "valid_count": valid_count,
        "valid_rate": (valid_count / total) if total else 0.0,
        "by_doctype": dict(sorted(by_doctype.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
        "field_fail_counts": dict(sorted(field_fail_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    return metrics, rows


def _write_artifacts(
    out_dir: Path,
    cfg: EvalConfig,
    images: List[Path],
    response: Dict[str, Any],
    metrics: Dict[str, Any],
    rows: List[Dict[str, Any]],
    started_at_utc: str,
    duration_s: float,
    per_image_latencies_ms: Optional[List[Dict[str, Any]]] = None,
) -> None:
    _safe_mkdir(out_dir)

    inputs = []
    for p in images:
        b = p.read_bytes()
        inputs.append(
            {
                "path": str(p),
                "filename": p.name,
                "sha256": _sha256_bytes(b),
                "bytes": len(b),
            }
        )

    metadata = {
        "started_at_utc": started_at_utc,
        "duration_s": duration_s,
        "name": cfg.name,
        "gateway_url": cfg.gateway_url,
        "endpoint": cfg.endpoint,
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "pip_freeze": _pip_freeze(),
        "inputs": inputs,
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if per_image_latencies_ms:
        (out_dir / "latencies.json").write_text(json.dumps(per_image_latencies_ms, indent=2), encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["filename", "ok", "doctype", "is_valid", "failure_reason", "message", "error"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --- Runner ---
def _call_batches_for_doctype(
    cfg: EvalConfig, images: List[Path], doc_type: Optional[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns combined results list and per-image latency records.
    """
    url = f"{cfg.gateway_url.rstrip('/')}{cfg.endpoint}"
    batches = _chunk(images, cfg.batch_size)
    combined_results: List[Dict[str, Any]] = []
    per_image_latencies: List[Dict[str, Any]] = []

    for batch in batches:
        params = {"doc_type": doc_type} if doc_type else None
        t0 = time.time()
        try:
            resp = _post_extract_batch_multipart(batch, url=url, timeout_s=cfg.timeout_s, params=params)
        except requests.exceptions.HTTPError as e:
            # If server 422s, dump body for diagnosis and attempt base64 fallback if enabled.
            resp_text = None
            try:
                resp_text = e.response.text
            except Exception:
                resp_text = str(e)
            print(f"HTTPError during multipart post: {e} - response: {resp_text}", file=sys.stderr)
            if not cfg.allow_base64_fallback:
                # convert to a minimal response object for consistent downstream handling
                combined_results.extend([{"filename": p.name, "ok": False, "error": "http_error"} for p in batch])
                continue
            try:
                resp = _post_extract_batch_base64(batch, url=url, timeout_s=cfg.timeout_s, params=params)
            except Exception as e2:
                print(f"Base64 fallback also failed: {e2}", file=sys.stderr)
                combined_results.extend([{"filename": p.name, "ok": False, "error": "fallback_failed"} for p in batch])
                continue
        except Exception as e:
            print(f"Transport error for batch: {e}", file=sys.stderr)
            combined_results.extend([{"filename": p.name, "ok": False, "error": "transport_error"} for p in batch])
            continue

        t1 = time.time()
        batch_dt = t1 - t0
        # resp is expected to be dict with "results": [...]
        batch_results = resp.get("results", []) if isinstance(resp, dict) else []
        combined_results.extend(batch_results)

        # estimate per-image latency
        per_image_ms = (batch_dt / max(1, len(batch))) * 1000.0
        for p in batch:
            per_image_latencies.append({"filename": p.name, "latency_ms": per_image_ms, "doc_type": doc_type or "unspecified"})

        # optional small pause between batches to avoid overloading
        if cfg.batch_delay_s and cfg.batch_delay_s > 0:
            time.sleep(cfg.batch_delay_s)

    return combined_results, per_image_latencies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, help="Folder containing images (recursively).")
    ap.add_argument("--gateway", default="http://127.0.0.1:8000", help="Gateway base URL.")
    ap.add_argument("--endpoint", default="/extract/batch", help="Batch extract endpoint path.")
    ap.add_argument("--name", default="gateway_eval", help="Run name suffix.")
    ap.add_argument("--timeout-s", type=int, default=60, help="HTTP timeout per batch (s).")
    ap.add_argument("--max-images", type=int, default=0, help="Cap number of images (0 = no cap).")
    ap.add_argument("--no-base64-fallback", action="store_true", help="Disable base64 fallback.")
    ap.add_argument("--batch-size", type=int, default=4, help="Number of images per batch to send.")
    ap.add_argument("--batch-delay-s", type=float, default=0.0, help="Sleep seconds between batches.")
    ap.add_argument("--doc-type", type=str, default=None, help="Optional doc_type to force for all requests.")
    args = ap.parse_args()

    cfg = EvalConfig(
        gateway_url=args.gateway.rstrip("/"),
        endpoint=args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}",
        test_dir=Path(args.test_dir),
        name=args.name,
        timeout_s=int(args.timeout_s),
        max_images=(None if int(args.max_images) <= 0 else int(args.max_images)),
        allow_base64_fallback=(not bool(args.no_base64_fallback)),
        batch_size=int(args.batch_size),
        batch_delay_s=float(args.batch_delay_s),
        doc_type=(args.doc_type.strip() if args.doc_type else None),
    )

    # choose strategy: if test_dir contains subdirs pan/aadhaar(aadhar) iterate them
    base = cfg.test_dir
    doc_dirs = {}
    if (base / "pan").is_dir():
        doc_dirs["pan"] = base / "pan"
    if (base / "aadhar").is_dir():
        doc_dirs["aadhar"] = base / "aadhar"
    if (base / "aadhaar").is_dir():
        doc_dirs["aadhar"] = base / "aadhaar"  # tolerate spelling

    combined_results: List[Dict[str, Any]] = []
    combined_latencies: List[Dict[str, Any]] = []
    all_images: List[Path] = []

    if doc_dirs and not cfg.doc_type:
        # iterate per doc type subdir
        for dt, path in doc_dirs.items():
            imgs = _iter_images(path)
            if cfg.max_images is not None:
                imgs = imgs[: cfg.max_images]
            if not imgs:
                continue
            all_images.extend(imgs)
            results, lat = _call_batches_for_doctype(cfg, imgs, doc_type=dt)
            combined_results.extend(results)
            combined_latencies.extend(lat)
    else:
        # single-shot on cfg.test_dir; use forced doc_type if given
        imgs = _iter_images(cfg.test_dir)
        if cfg.max_images is not None:
            imgs = imgs[: cfg.max_images]
        if not imgs:
            raise SystemExit(f"No images found under: {cfg.test_dir}")
        all_images.extend(imgs)
        results, lat = _call_batches_for_doctype(cfg, imgs, doc_type=cfg.doc_type)
        combined_results.extend(results)
        combined_latencies.extend(lat)

    # collapse into response shape expected by downstream functions
    response = {"results": combined_results}

    started = _utc_ts_compact()
    out_dir = Path("artifacts") / "eval_runs" / f"{started}_{cfg.name}"
    t_total = 0.0  # compute aggregated total time as sum of latencies approximations
    if combined_latencies:
        # sum of per-image latencies (ms) -> seconds
        t_total = sum([r["latency_ms"] for r in combined_latencies]) / 1000.0

    metrics, rows = _compute_metrics(response)
    _write_artifacts(
        out_dir=out_dir,
        cfg=cfg,
        images=all_images,
        response=response,
        metrics=metrics,
        rows=rows,
        started_at_utc=started,
        duration_s=t_total,
        per_image_latencies_ms=combined_latencies,
    )

    print(json.dumps({"out_dir": str(out_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
