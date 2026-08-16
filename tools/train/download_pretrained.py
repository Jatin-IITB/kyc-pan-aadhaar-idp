"""Download pre-trained YOLO field detection models for PAN and Aadhaar.

Usage:
    python -m tools.train.download_pretrained --type aadhaar
    python -m tools.train.download_pretrained --type pan --roboflow-key YOUR_KEY
    python -m tools.train.download_pretrained --type all --roboflow-key YOUR_KEY

Aadhaar: Downloads arnabdhar/YOLOv8-nano-aadhar-card from HuggingFace (no auth needed).
PAN:     Downloads pancard-info-detection dataset from Roboflow, then trains YOLOv8n.
         Requires a free Roboflow API key (universe.roboflow.com → Settings → API Key).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models" / "yolov8"


def download_aadhaar() -> Path:
    """Download pre-trained Aadhaar field detector from HuggingFace."""
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_id = "arnabdhar/YOLOv8-nano-aadhar-card"
    print(f"Downloading {repo_id} from HuggingFace...")

    pt_files = [f for f in list_repo_files(repo_id) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {repo_id}")
    pt_path = hf_hub_download(repo_id, pt_files[0])
    dst_dir = MODELS_DIR / "aadhar_field_detector_v1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy2(pt_path, dst)

    from ultralytics import YOLO
    model = YOLO(str(dst))
    print(f"  Classes: {model.names}")
    print(f"  Saved to: {dst}")
    return dst


def _ssl_context():
    """TLS context tolerant of corporate MITM proxies (Zscaler, Forcepoint, etc.).

    Python 3.13+ enables VERIFY_X509_STRICT by default, which rejects many
    corporate root CAs because their Basic Constraints extension is not marked
    critical (an RFC 5280 violation these appliances commonly ship).

    Clearing that one flag keeps FULL chain-of-trust and hostname verification
    against the system trust store — it only relaxes the strict RFC conformance
    check. Verification is NOT disabled.

    Set SSL_CERT_FILE to point at a custom CA bundle if you need one.
    """
    import ssl

    cafile = os.environ.get("SSL_CERT_FILE")
    ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def fetch_roboflow_dataset(workspace: str, project: str, version: int,
                           api_key: str, dest: Path, fmt: str = "yolov8") -> Path:
    """Download a Roboflow dataset via the REST API (no SDK dependency).

    Avoids `pip install roboflow`, which pins opencv-python-headless and would
    conflict with the project's existing OpenCV builds.
    """
    import json
    import shutil as _shutil
    import urllib.request
    import zipfile

    ctx = _ssl_context()
    api_url = (f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}"
               f"?api_key={api_key}")
    print(f"  Requesting export link for {workspace}/{project} v{version}...")
    with urllib.request.urlopen(api_url, timeout=60, context=ctx) as resp:
        payload = json.load(resp)

    link = payload.get("export", {}).get("link")
    if not link:
        raise SystemExit(f"Roboflow API returned no download link: {payload}")

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "dataset.zip"
    print(f"  Downloading dataset archive...")
    with urllib.request.urlopen(link, timeout=600, context=ctx) as resp, \
            open(zip_path, "wb") as fh:
        _shutil.copyfileobj(resp, fh)

    print(f"  Extracting to {dest}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()

    data_yaml = dest / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"No data.yaml found in extracted dataset at {dest}")
    return data_yaml


def download_and_train_pan(api_key: str, epochs: int = 50, imgsz: int = 640) -> Path:
    """Download PAN card dataset from Roboflow and train YOLOv8n."""
    print("Downloading PAN card dataset from Roboflow...")
    data_yaml = fetch_roboflow_dataset(
        workspace="documentverification-23azt",
        project="pancard-info-detection",
        version=1,
        api_key=api_key,
        dest=REPO_ROOT / "data" / "pan_roboflow",
    )

    print(f"  Dataset ready: {data_yaml}")
    print(f"  Training YOLOv8n for {epochs} epochs...")

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name="pan_fields",
        project=str(REPO_ROOT / "data" / "yolo_runs"),
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
    )

    run_dir = REPO_ROOT / "data" / "yolo_runs" / "pan_fields"
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    dst_dir = MODELS_DIR / "pan_field_detector_v1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy2(best_pt, dst)

    model = YOLO(str(dst))
    print(f"  Classes: {model.names}")
    print(f"  Saved to: {dst}")

    metrics = model.val(data=str(data_yaml), verbose=False)
    print(f"  mAP50={metrics.box.map50:.3f}, mAP50-95={metrics.box.map:.3f}")
    return dst


def train_pan_from_local(data_yaml: str, epochs: int = 50, imgsz: int = 640) -> Path:
    """Train PAN detector from a locally downloaded YOLO-format dataset."""
    from ultralytics import YOLO

    print(f"Training YOLOv8n on {data_yaml} for {epochs} epochs...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name="pan_fields",
        project=str(REPO_ROOT / "data" / "yolo_runs"),
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
    )

    run_dir = REPO_ROOT / "data" / "yolo_runs" / "pan_fields"
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    dst_dir = MODELS_DIR / "pan_field_detector_v1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy2(best_pt, dst)

    model = YOLO(str(dst))
    print(f"  Classes: {model.names}")
    print(f"  Saved to: {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["aadhaar", "pan", "all"], default="all")
    ap.add_argument("--roboflow-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                    help="Roboflow API key. Prefer the ROBOFLOW_API_KEY env var so the "
                         "key stays out of shell history.")
    ap.add_argument("--pan-data-yaml", help="Path to local YOLO-format dataset.yaml for PAN "
                    "(alternative to Roboflow download)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    types = ["aadhaar", "pan"] if args.type == "all" else [args.type]

    for t in types:
        print(f"\n{'='*60}")
        print(f"  {t.upper()}")
        print(f"{'='*60}\n")

        if t == "aadhaar":
            download_aadhaar()
        elif t == "pan":
            if args.pan_data_yaml:
                train_pan_from_local(args.pan_data_yaml, args.epochs, args.imgsz)
            elif args.roboflow_key:
                download_and_train_pan(args.roboflow_key, args.epochs, args.imgsz)
            else:
                print("  PAN requires either --roboflow-key or --pan-data-yaml.")
                print("  Get a free Roboflow API key at: https://universe.roboflow.com")
                print("  Or download a YOLO-format PAN dataset manually and pass --pan-data-yaml")


if __name__ == "__main__":
    main()
