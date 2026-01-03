from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LinkMode = Literal["copy", "hardlink", "symlink"]


@dataclass(frozen=True)
class Candidate:
    path: Path
    doc_type: str


def _iter_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _post_extract_batch_async(
    gateway: str,
    doc_type: str,
    paths: List[Path],
    timeout_s: int,
) -> Dict[str, Any]:
    """
    Submits to /batches, polls for completion.
    On timeout, returns partial results rather than crashing.
    """
    # FIX 1: Hardcoded endpoint for safety
    url = f"{gateway.rstrip('/')}/batches"
    
    files = []
    opened = []
    
    try:
        for p in paths:
            fh = open(p, "rb")
            opened.append(fh)
            # FIX 3: Filename mapping relies on p.name (acceptable for now per guidelines)
            files.append(("files", (p.name, fh, "application/octet-stream")))

        print(f"  -> Submitting {len(paths)} images to {url}...")
        r = requests.post(
            url, 
            params={"doc_type": doc_type}, 
            files=files, 
            timeout=30 
        )
        r.raise_for_status()
        batch_data = r.json()
        batch_id = batch_data["batch_id"]
        print(f"  -> Batch {batch_id} accepted. Polling...")

        # Poll loop
        start_t = time.time()
        last_log_t = 0.0
        is_tty = sys.stdout.isatty()
        
        status_data = {}

        while True:
            # FIX: Timeout semantics - break, don't crash
            if time.time() - start_t > timeout_s:
                print(f"\n  [WARN] Batch {batch_id} timed out after {timeout_s}s. Fetching partials...")
                break

            try:
                poll_r = requests.get(f"{gateway.rstrip('/')}/batches/{batch_id}", timeout=10)
                poll_r.raise_for_status()
                status_data = poll_r.json()
            except Exception as e:
                print(f"\n  [WARN] Poll failed: {e}. Retrying...")
                time.sleep(2)
                continue
            
            status = status_data.get("status", "UNKNOWN")
            summary = status_data.get("summary", {})
            
            done = summary.get("SUCCEEDED", 0) + summary.get("FAILED", 0)
            total = len(paths)

            # FIX: CI-friendly logging
            if is_tty:
                print(f"     Status: {status} [{done}/{total}]", end="\r")
            else:
                # In CI, log only every 30s to avoid noise
                if time.time() - last_log_t > 30:
                    print(f"     Status: {status} [{done}/{total}]")
                    last_log_t = time.time()

            if status == "COMPLETED":
                if is_tty:
                    print() # Newline after \r
                print(f"  -> Batch completed.")
                break
            
            time.sleep(2)

        # Fetch results (full or partial)
        final_results = []
        jobs_list = status_data.get("jobs", [])
        
        # If timeout occurred, jobs_list might be incomplete or contain RUNNING items.
        # We process whatever we have.
        for job_summary in jobs_list:
            # Optimistic: if result is already embedded
            if "result" in job_summary or "error" in job_summary:
                final_results.append(job_summary)
                continue

            # Fetch details
            job_id = job_summary.get("job_id")
            if not job_id:
                continue

            try:
                job_r = requests.get(f"{gateway.rstrip('/')}/jobs/{job_id}", timeout=10)
                if job_r.status_code == 200:
                    final_results.append(job_r.json())
                else:
                    final_results.append({
                        "filename": job_summary.get("filename"),
                        "ok": False, 
                        "error": f"fetch_error_{job_r.status_code}"
                    })
            except Exception as e:
                final_results.append({
                    "filename": job_summary.get("filename"),
                    "ok": False, 
                    "error": f"fetch_exception_{str(e)}"
                })

        return {"count": len(final_results), "results": final_results}

    finally:
        for fh in opened:
            try:
                fh.close()
            except Exception:
                pass


def _chunk(xs: List[Path], batch_size: int) -> List[List[Path]]:
    return [xs[i : i + batch_size] for i in range(0, len(xs), batch_size)]


def _get_is_valid(item: Dict[str, Any]) -> bool:
    if not item.get("ok", False):
        return False
    res = item.get("result") or {}
    val = res.get("validation") or {}
    return bool(val.get("is_valid", False))


def _get_message(item: Dict[str, Any]) -> str:
    if not item.get("ok", False):
        return str(item.get("error", ""))
    res = item.get("result") or {}
    val = res.get("validation") or {}
    return str(val.get("message", ""))


def _get_doc_type(item: Dict[str, Any]) -> str:
    if not item.get("ok", False):
        return "unknown"
    res = item.get("result") or {}
    dt = res.get("document_type")
    return dt if isinstance(dt, str) and dt else "unknown"


def _materialize(src: Path, dst: Path, mode: LinkMode) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        os.link(src, dst)
        return

    os.symlink(src, dst)


def _stable_sample(paths: List[Path], k: int, seed: int) -> List[Path]:
    rng = random.Random(seed)
    keyed = [(p, _sha256_file(p)) for p in paths]
    rng.shuffle(keyed)
    keyed.sort(key=lambda t: t[1])
    return [p for p, _ in keyed[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8000")
    # FIX: Removed --endpoint argument
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)

    ap.add_argument("--pan-root", default="data/processed/pan/images/test")
    ap.add_argument("--aadhaar-root", default="data/processed/aadhar/images/test")

    ap.add_argument("--out-root", default="data/test_cases")
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")

    ap.add_argument("--golden-pan", type=int, default=50)
    ap.add_argument("--golden-aadhaar", type=int, default=50)
    ap.add_argument("--hard-pan", type=int, default=50)
    ap.add_argument("--hard-aadhaar", type=int, default=50)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    gateway = args.gateway.rstrip("/")

    candidates: List[Candidate] = []
    if Path(args.pan_root).exists():
        for p in _iter_images(Path(args.pan_root)):
            candidates.append(Candidate(path=p, doc_type="pan"))
    if Path(args.aadhaar_root).exists():
        for p in _iter_images(Path(args.aadhaar_root)):
            candidates.append(Candidate(path=p, doc_type="aadhar"))

    if not candidates:
        raise SystemExit("No candidate images found.")

    by_type: Dict[str, List[Path]] = {"pan": [], "aadhar": []}
    for c in candidates:
        by_type[c.doc_type].append(c.path)

    all_rows: List[Dict[str, Any]] = []

    for dt, paths in by_type.items():
        if not paths:
            continue
            
        for batch in _chunk(paths, args.batch_size):
            resp = _post_extract_batch_async(
                gateway=gateway,
                # endpoint removed
                doc_type=dt,
                paths=batch,
                timeout_s=args.timeout_s,
            )
            
            for item in resp.get("results", []):
                fname = item.get("filename")
                ok = bool(item.get("ok", False))
                is_valid = _get_is_valid(item)
                msg = _get_message(item)
                out_dt = _get_doc_type(item)

                src = next((p for p in batch if p.name == fname), None)

                all_rows.append(
                    {
                        "src_path": str(src) if src else "",
                        "filename": str(fname),
                        "requested_doc_type": dt,
                        "result_doc_type": out_dt,
                        "ok": ok,
                        "is_valid": is_valid,
                        "message": msg,
                    }
                )

    pan_rows = [r for r in all_rows if r["requested_doc_type"] == "pan" and r["src_path"]]
    aad_rows = [r for r in all_rows if r["requested_doc_type"] == "aadhar" and r["src_path"]]

    pan_golden = [Path(r["src_path"]) for r in pan_rows if r["ok"] and r["is_valid"]]
    aad_golden = [Path(r["src_path"]) for r in aad_rows if r["ok"] and r["is_valid"]]

    pan_hard = [Path(r["src_path"]) for r in pan_rows if (not r["ok"]) or (not r["is_valid"])]
    aad_hard = [Path(r["src_path"]) for r in aad_rows if (not r["ok"]) or (not r["is_valid"])]

    pan_golden = _stable_sample(pan_golden, args.golden_pan, args.seed)
    aad_golden = _stable_sample(aad_golden, args.golden_aadhaar, args.seed + 1)
    pan_hard = _stable_sample(pan_hard, args.hard_pan, args.seed + 2)
    aad_hard = _stable_sample(aad_hard, args.hard_aadhaar, args.seed + 3)

    out_root = Path(args.out_root)
    golden_pan_dir = out_root / "golden" / "pan"
    golden_aad_dir = out_root / "golden" / "aadhaar"
    hard_pan_dir = out_root / "hard" / "pan"
    hard_aad_dir = out_root / "hard" / "aadhaar"

    for p in pan_golden:
        _materialize(p, golden_pan_dir / p.name, args.mode)
    for p in aad_golden:
        _materialize(p, golden_aad_dir / p.name, args.mode)

    for p in pan_hard:
        _materialize(p, hard_pan_dir / p.name, args.mode)
    for p in aad_hard:
        _materialize(p, hard_aad_dir / p.name, args.mode)

    if args.write_manifest:
        man_dir = out_root / "_manifests"
        man_dir.mkdir(parents=True, exist_ok=True)
        (man_dir / "suite_build_results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

        with (man_dir / "suite_build_results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["src_path", "filename", "requested_doc_type", "result_doc_type", "ok", "is_valid", "message"],
            )
            w.writeheader()
            w.writerows(all_rows)

    print("Done.")
    print(f"golden/pan={len(pan_golden)} golden/aadhaar={len(aad_golden)}")
    print(f"hard/pan={len(pan_hard)} hard/aadhaar={len(aad_hard)}")


if __name__ == "__main__":
    main()
