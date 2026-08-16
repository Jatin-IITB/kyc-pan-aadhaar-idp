"""Train YOLOv8n field detectors on real-world annotated data.

Data sources (all free):
  - Roboflow Universe: pancard-info-detection, AADHAAR-CARD-DETAILS
  - Kaggle: Indian ID card datasets with YOLO annotations
  - HuggingFace: arnabdhar/YOLOv8-nano-aadhar-card (pre-trained, no training needed)

Usage:
    # Train PAN detector on Roboflow data (needs free API key)
    python -m tools.train.train_yolo_fields --type pan --roboflow-key YOUR_KEY

    # Train on a locally downloaded YOLO-format dataset
    python -m tools.train.train_yolo_fields --type pan --data-yaml path/to/data.yaml

    # Evaluate an existing model
    python -m tools.train.train_yolo_fields --type pan --eval-only

For Aadhaar, use the pre-trained HuggingFace model instead:
    python -m tools.train.download_pretrained --type aadhaar
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models" / "yolov8"
DATA_DIR = REPO_ROOT / "data"

ROBOFLOW_DATASETS = {
    "pan": {
        "workspace": "documentverification-23azt",
        "project": "pancard-info-detection",
        "version": 1,
    },
    "aadhaar": {
        "workspace": "cutm-vmq1p",
        "project": "aadhaar-card-details-v2ict",
        "version": 1,
    },
}


def download_roboflow(doc_type: str, api_key: str) -> Path:
    """Download via Roboflow's REST API — no SDK dependency.

    The roboflow SDK pins opencv-python-headless, which conflicts with the
    project's existing OpenCV builds. See tools/train/download_pretrained.py.
    """
    from tools.train.download_pretrained import fetch_roboflow_dataset

    info = ROBOFLOW_DATASETS[doc_type]
    data_yaml = fetch_roboflow_dataset(
        workspace=info["workspace"],
        project=info["project"],
        version=info["version"],
        api_key=api_key,
        dest=DATA_DIR / f"{doc_type}_roboflow",
    )
    print(f"  Downloaded to: {data_yaml.parent}")
    return data_yaml


def train(doc_type: str, data_yaml: Path, epochs: int, imgsz: int,
          batch: int) -> Path:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    run_name = f"{doc_type}_fields"

    print(f"  Training YOLOv8n on {data_yaml} ({epochs} epochs, imgsz={imgsz})...")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project=str(DATA_DIR / "yolo_runs"),
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
    )

    run_dir = DATA_DIR / "yolo_runs" / run_name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    dst_dir = MODELS_DIR / f"{doc_type}_field_detector_v1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy2(best_pt, dst)
    print(f"  Model saved to: {dst}")
    return dst


def evaluate(doc_type: str, data_yaml: Path | None = None) -> dict:
    from ultralytics import YOLO

    weights = MODELS_DIR / f"{doc_type}_field_detector_v1" / "best.pt"
    if not weights.exists():
        print(f"  No model found at {weights}")
        return {}

    model = YOLO(str(weights))
    print(f"  Classes: {model.names}")

    if data_yaml and data_yaml.exists():
        metrics = model.val(data=str(data_yaml), verbose=False)
        result = {
            "doc_type": doc_type,
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        print(f"  mAP50={result['mAP50']:.3f}, mAP50-95={result['mAP50-95']:.3f}, "
              f"P={result['precision']:.3f}, R={result['recall']:.3f}")
        return result

    print("  No data.yaml provided — skipping validation metrics")
    return {"doc_type": doc_type, "classes": model.names}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["pan", "aadhaar"], required=True)
    ap.add_argument("--roboflow-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                    help="Roboflow API key. Prefer the ROBOFLOW_API_KEY env var so the "
                         "key stays out of shell history.")
    ap.add_argument("--data-yaml", help="Path to local YOLO-format dataset.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-only", action="store_true",
                    help="Only evaluate existing model, skip training")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  {args.type.upper()} FIELD DETECTOR")
    print(f"{'='*60}\n")

    if args.eval_only:
        data_yaml = Path(args.data_yaml) if args.data_yaml else None
        evaluate(args.type, data_yaml)
        return

    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    elif args.roboflow_key:
        if args.type not in ROBOFLOW_DATASETS:
            print(f"  No Roboflow dataset configured for {args.type}")
            return
        print("[1/3] Downloading dataset from Roboflow...")
        data_yaml = download_roboflow(args.type, args.roboflow_key)
    else:
        print("  Provide --roboflow-key or --data-yaml to supply training data.")
        print(f"  For Aadhaar, use the pre-trained model instead:")
        print(f"    python -m tools.train.download_pretrained --type aadhaar")
        return

    print(f"\n[2/3] Training...")
    weights = train(args.type, data_yaml, args.epochs, args.imgsz, args.batch)

    print(f"\n[3/3] Evaluating...")
    evaluate(args.type, data_yaml)


if __name__ == "__main__":
    main()
