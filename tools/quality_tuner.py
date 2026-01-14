#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import random

import cv2
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# project root
sys.path.append(str(Path(__file__).resolve().parents[2]))
from services.preprocessing.quality import check_image_quality

console = Console()

DATA_POOLS = {
    "aadhaar": Path("data/processed/aadhar/images"),
    "pan": Path("data/processed/pan/images"),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(root: Path, max_samples: int | None):
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if max_samples and len(imgs) > max_samples:
        imgs = random.sample(imgs, max_samples)
    return imgs


def scan_pool(doc_type: str, img_paths):
    rows = []
    for p in tqdm(img_paths, desc=f"Scanning {doc_type}"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        is_good, m = check_image_quality(img)
        rows.append({
            "doc_type": doc_type,
            "file": p.name,
            "pass": bool(is_good),
            "blur": float(m.get("blur_score", 0)),
            "white": float(m.get("overexposed_ratio", 0)),
            "black": float(m.get("underexposed_ratio", 0)),
        })
    return rows


def recommend(df, keep=0.95):
    pass_df = df[df["pass"]]
    if pass_df.empty:
        return None

    return {
        "min_blur_score": round(pass_df["blur"].quantile(1 - keep), 2),
        "max_white_ratio": round(pass_df["white"].quantile(keep), 4),
        "max_black_ratio": round(pass_df["black"].quantile(keep), 4),
    }


def print_percentiles(df, col):
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    table = Table(show_header=True)
    table.add_column("Pctl")
    table.add_column(col)
    for q in qs:
        table.add_row(f"{int(q*100)}%", f"{df[col].quantile(q):.4f}")
    console.print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-pool", type=int, default=3000,
                    help="Cap images per dataset (for speed)")
    ap.add_argument("--keep-fraction", type=float, default=0.95)
    ap.add_argument("--out", default="quality_scan_processed.csv")
    args = ap.parse_args()

    all_rows = []

    for doc, root in DATA_POOLS.items():
        if not root.exists():
            console.print(f"[yellow]Skipping missing pool {root}[/yellow]")
            continue
        imgs = iter_images(root, args.max_per_pool)
        all_rows.extend(scan_pool(doc, imgs))

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)

    console.print(f"\n[green]Saved scan → {args.out}[/green]")
    console.print(df.groupby(["doc_type", "pass"]).size())

    console.print("\n[bold]Blur percentiles (PASS set):[/bold]")
    print_percentiles(df[df["pass"]], "blur")

    console.print("\n[bold]White ratio percentiles (PASS set):[/bold]")
    print_percentiles(df[df["pass"]], "white")

    console.print("\n[bold]Black ratio percentiles (PASS set):[/bold]")
    print_percentiles(df[df["pass"]], "black")

    rec = recommend(df, keep=args.keep_fraction)
    if rec:
        console.print("\n[bold green]Recommended thresholds:[/bold green]")
        console.print(rec)
        console.print("\nPaste into config/thresholds.yaml:\n")
        console.print("quality_gate:")
        for k, v in rec.items():
            console.print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
