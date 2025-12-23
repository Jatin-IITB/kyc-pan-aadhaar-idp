"""
Dataset Generation CLI
======================

Generate synthetic PAN/Aadhaar training dataset with ground truth annotations.

Usage:
    python -m tools.synthetic_id_generator.generate_dataset \\
        --output data/synthetic \\
        --num-pan 100 \\
        --num-aadhaar 100 \\
        --seed 42

Output structure:
    data/synthetic/
    ├── images/
    │   ├── 000001_PAN.jpg
    │   ├── 000002_AADHAAR.jpg
    │   └── ...
    ├── labels/
    │   ├── 000001_PAN.json
    │   ├── 000002_AADHAAR.json
    │   └── ...
    └── manifest.csv
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any
import cv2
import pandas as pd
from tqdm import tqdm

from .config import GeneratorConfig, PANConfig, AadhaarConfig
from .pan_generator import PANCardGenerator
from .aadhaar_generator import AadhaarCardGenerator


def save_sample(
    result: Dict[str, Any],
    output_id: str,
    images_dir: Path,
    labels_dir: Path,
) -> Dict[str, Any]:
    """
    Save generated card image and ground truth label.
    
    Returns manifest row dict.
    """
    card_img = result["card_image"]
    ground_truth = result["ground_truth"]
    doc_type = ground_truth["doc_type"]
    
    # Save image
    img_filename = f"{output_id}_{doc_type}.jpg"
    img_path = images_dir / img_filename
    cv2.imwrite(str(img_path), card_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Save label (JSON)
    label_filename = f"{output_id}_{doc_type}.json"
    label_path = labels_dir / label_filename
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    # Manifest row
    fields = ground_truth["fields"]
    return {
        "id": output_id,
        "doc_type": doc_type,
        "image_filename": img_filename,
        "label_filename": label_filename,
        "acquisition_style": ground_truth["acquisition_style"],
        **fields,  # Flatten all fields into manifest
    }


def generate_dataset(
    num_pan: int = 100,
    num_aadhaar: int = 100,
    output_dir: Path = Path("data/synthetic"),
    seed: int | None = None,
) -> None:
    """
    Generate complete synthetic dataset.
    
    Args:
        num_pan: Number of PAN cards to generate
        num_aadhaar: Number of Aadhaar cards to generate
        output_dir: Root output directory
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    # Setup directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir.resolve()}")
    print(f"🎯 Target: {num_pan} PAN + {num_aadhaar} Aadhaar cards\n")
    
    # Initialize generators
    config = GeneratorConfig(output_dir=output_dir)
    pan_gen = PANCardGenerator(config, PANConfig())
    aadhaar_gen = AadhaarCardGenerator(config, AadhaarConfig())
    
    manifest_rows = []
    counter = 1
    
    # Generate PAN cards
    print("🏦 Generating PAN cards...")
    for i in tqdm(range(num_pan), desc="PAN", unit="card"):
        # Vary status codes (mostly P=Person, some others)
        status_code = random.choices(
            ["P", "C", "H", "F", "A", "T"],
            weights=[0.85, 0.05, 0.04, 0.03, 0.02, 0.01],
            k=1
        )[0]
        
        output_id = f"{counter:06d}"
        result = pan_gen.generate(status_code=status_code, output_id=output_id)
        
        row = save_sample(result, output_id, images_dir, labels_dir)
        manifest_rows.append(row)
        
        counter += 1
    
    # Generate Aadhaar cards
    print("\n🆔 Generating Aadhaar cards...")
    for i in tqdm(range(num_aadhaar), desc="Aadhaar", unit="card"):
        output_id = f"{counter:06d}"
        result = aadhaar_gen.generate(output_id=output_id)
        
        row = save_sample(result, output_id, images_dir, labels_dir)
        manifest_rows.append(row)
        
        counter += 1
    
    # Save manifest CSV
    print("\n📋 Saving manifest...")
    manifest_path = output_dir / "manifest.csv"
    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    
    print(f"\n✅ Dataset generation complete!")
    print(f"   Images: {len(list(images_dir.glob('*.jpg')))}")
    print(f"   Labels: {len(list(labels_dir.glob('*.json')))}")
    print(f"   Manifest: {manifest_path}")
    
    # Distribution stats
    print("\n📊 Distribution:")
    print(df["doc_type"].value_counts())
    print("\n📷 Acquisition styles:")
    print(df["acquisition_style"].value_counts())
    
    if "status_code" in df.columns:
        print("\n🏷️  PAN status codes:")
        print(df[df["doc_type"] == "PAN"]["status_code"].value_counts())


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic PAN/Aadhaar training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory for generated dataset",
    )
    
    parser.add_argument(
        "--num-pan",
        type=int,
        default=100,
        help="Number of PAN cards to generate",
    )
    
    parser.add_argument(
        "--num-aadhaar",
        type=int,
        default=100,
        help="Number of Aadhaar cards to generate",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        num_pan=args.num_pan,
        num_aadhaar=args.num_aadhaar,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
