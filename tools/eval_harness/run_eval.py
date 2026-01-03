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
    test_dir: Path
    name: str
    timeout_s: int
    max_images: Optional[int]
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


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _iter_images(test_dir: Path) -> List[Path]:
    imgs = [p for p in sorted(test_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return imgs


def _chunk(xs: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        return [xs]
    return [xs[i : i + batch_size] for i in range(0, len(xs), batch_size)]


# --- HTTP Async helpers (Refactored) ---
def _submit_and_poll_batch(
    images: List[Path], 
    gateway_url: str, 
    timeout_s: int, 
    params: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Submits a batch, polls for completion, and fetches results.
    Returns: (list_of_results, total_batch_duration_seconds)
    """
    submit_url = f"{gateway_url.rstrip('/')}/batches"
    
    files = []
    opened = []
    
    t0 = time.time()
    try:
        # 1. Prepare Upload
        for p in images:
            f = open(p, "rb")
            opened.append(f)
            # Filename is key for mapping back results
            files.append(("files", (p.name, f, "application/octet-stream")))

        # 2. Submit
        r = requests.post(submit_url, files=files, params=params, timeout=30)
        r.raise_for_status()
        batch_id = r.json()["batch_id"]
        
        # 3. Poll
        while True:
            elapsed = time.time() - t0
            if elapsed > timeout_s:
                print(f" [WARN] Batch {batch_id} timed out after {elapsed:.1f}s. Fetching partials...", file=sys.stderr)
                break
            
            # Check status
            poll_r = requests.get(f"{gateway_url.rstrip('/')}/batches/{batch_id}", timeout=10)
            poll_r.raise_for_status()
            status_data = poll_r.json()
            status = status_data.get("status")
            
            if status == "COMPLETED":
                break
            
            # Exponential backoff cap at 2s
            time.sleep(min(elapsed * 0.1 + 0.5, 2.0))

        # 4. Fetch Results
        final_results = []
        
        # 'jobs' list from /batches/{id} contains statuses. We need full results.
        jobs_summary = status_data.get("jobs", [])
        
        for job_meta in jobs_summary:
            job_id = job_meta.get("job_id")
            fname = job_meta.get("filename", "unknown")
            
            # If result is somehow embedded (future optimization), use it
            if "result" in job_meta:
                 final_results.append(job_meta)
                 continue
                 
            # Fetch single job details
            try:
                job_r = requests.get(f"{gateway_url.rstrip('/')}/jobs/{job_id}", timeout=10)
                if job_r.status_code == 200:
                    final_results.append(job_r.json())
                else:
                    final_results.append({
                        "filename": fname,
                        "ok": False,
                        "error": f"fetch_error_{job_r.status_code}"
                    })
            except Exception as e:
                final_results.append({
                    "filename": fname,
                    "ok": False,
                    "error": f"fetch_exception_{str(e)}"
                })
        
        # Calculate total duration for this batch
        duration = time.time() - t0
        return final_results, duration

    finally:
        for f in opened:
            try:
                f.close()
            except Exception:
                pass


# --- Result parsing helpers (Unchanged) ---
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
        "endpoint": "/batches", # Hardcoded as we enforce async
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
    batches = _chunk(images, cfg.batch_size)
    combined_results: List[Dict[str, Any]] = []
    per_image_latencies: List[Dict[str, Any]] = []

    print(f"Processing {len(images)} images in {len(batches)} batches (async)...")

    for i, batch in enumerate(batches):
        params = {"doc_type": doc_type} if doc_type else None
        
        try:
            # Call new ASYNC helper
            batch_results, batch_duration = _submit_and_poll_batch(
                batch, 
                gateway_url=cfg.gateway_url, 
                timeout_s=cfg.timeout_s, 
                params=params
            )
            
            combined_results.extend(batch_results)

            # Estimate per-image latency (Total Batch Time / N images)
            # This captures Queue Wait + Processing Time
            per_image_ms = (batch_duration / max(1, len(batch))) * 1000.0
            
            for p in batch:
                per_image_latencies.append({
                    "filename": p.name, 
                    "latency_ms": per_image_ms, 
                    "doc_type": doc_type or "unspecified"
                })
                
            print(f"  Batch {i+1}/{len(batches)} done. Avg Latency: {per_image_ms:.0f}ms/doc")

        except Exception as e:
            print(f"  Batch {i+1} failed completely: {e}", file=sys.stderr)
            # Fill with errors so we don't crash stats
            combined_results.extend([{"filename": p.name, "ok": False, "error": str(e)} for p in batch])

        # optional small pause
        if cfg.batch_delay_s and cfg.batch_delay_s > 0:
            time.sleep(cfg.batch_delay_s)

    return combined_results, per_image_latencies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, help="Folder containing images (recursively).")
    ap.add_argument("--out-dir", help="Explicit output directory.")
    ap.add_argument("--gateway", default="http://127.0.0.1:8000", help="Gateway base URL.")
    # endpoint arg removed as we enforce /batches
    ap.add_argument("--name", default="gateway_eval", help="Run name suffix.")
    ap.add_argument("--timeout-s", type=int, default=300, help="Timeout per batch (s).")
    ap.add_argument("--max-images", type=int, default=0, help="Cap number of images (0 = no cap).")
    ap.add_argument("--batch-size", type=int, default=16, help="Number of images per batch to send.")
    ap.add_argument("--batch-delay-s", type=float, default=0.0, help="Sleep seconds between batches.")
    ap.add_argument("--doc-type", type=str, default=None, help="Optional doc_type to force for all requests.")
    args = ap.parse_args()

    cfg = EvalConfig(
        gateway_url=args.gateway.rstrip("/"),
        test_dir=Path(args.test_dir),
        name=args.name,
        timeout_s=int(args.timeout_s),
        max_images=(None if int(args.max_images) <= 0 else int(args.max_images)),
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
    if args.out_dir:
         out_dir = Path(args.out_dir)
    else:
         out_dir = Path("artifacts") / "eval_runs" / f"{started}_{cfg.name}"
    t_total = 0.0 
    if combined_latencies:
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
