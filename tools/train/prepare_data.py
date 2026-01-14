# tools/train/prepare_data.py
import json
from pathlib import Path
from typing import Dict

# --- Configuration ---
GT_DIR = Path("data/reviewed_ground_truth")
OUTPUT_FILE = Path("data/training/dataset_v1.jsonl")

# Same image roots as in apps/review_ui/main.py
IMG_ROOTS = [
    Path("data/test_cases_async"),
    Path("data/processed/pan/images/test"),
    Path("data/processed/aadhar/images/test")
]

def find_image(filename: str) -> str:
    """Replicates FileSystemAdapter._find_image logic."""
    if not filename: return None
    for root in IMG_ROOTS:
        if not root.exists(): continue
        # Robust search handling subdirectories
        found = list(root.rglob(filename))
        if found:
            return str(found[0])
    return None

def infer_doc_type(data: Dict) -> str:
    """Heuristic to determine doc type from fields if missing."""
    keys = " ".join(data.keys()).lower()
    if "pan" in keys or "father" in keys:
        return "pan"
    if "aadhaar" in keys or "gender" in keys:
        return "aadhaar"
    return "unknown"

def main():
    print(f"🔍 Scanning {GT_DIR} for reviewed files...")
    if not GT_DIR.exists():
        print(f"❌ GT Directory not found: {GT_DIR}")
        return

    dataset = []
    
    # 1. Iterate over all Ground Truth JSONs
    gt_files = list(GT_DIR.glob("gt_*.json"))
    print(f"found {len(gt_files)} reviewed documents.")

    for gt_path in gt_files:
        try:
            # Load Review Result
            review = json.loads(gt_path.read_text(encoding="utf-8"))

            job_id = review.get("job_id")
            corrections = review.get("correction", {})
            
            # 2. Find the Source Image
            image_path = find_image(job_id)
            
            if not image_path:
                print(f"⚠️  Image not found for ID: {job_id} (Skipping)")
                continue

            # 3. Construct Training Record
            record = {
                "image_path": str(Path(image_path).resolve()),
                "filename": job_id,
                "document_type": infer_doc_type(corrections),
                "ground_truth": corrections,
                "metadata": {
                    "reviewer": review.get("reviewer"),
                    "reviewed_at": review.get("timestamp"),
                    "source_gt_file": str(gt_path.name)
                }
            }
            dataset.append(record)

        except Exception as e:
            print(f"❌ Error processing {gt_path.name}: {e}")

    # 4. Save to JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"\n✅ Successfully created dataset with {len(dataset)} records.")
    print(f"📁 Output: {OUTPUT_FILE}")
    
    # Stats
    doc_types = {}
    for d in dataset:
        dt = d["document_type"]
        doc_types[dt] = doc_types.get(dt, 0) + 1
    print("📊 Distribution:", doc_types)

if __name__ == "__main__":
    main()
