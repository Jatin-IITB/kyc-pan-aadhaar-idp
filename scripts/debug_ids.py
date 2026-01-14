import json
import os
from pathlib import Path

# Config
RESULTS_PATH = Path("runs/eval_async_v2/results.json")
GT_DIR = Path("data/reviewed_ground_truth")

print("--- DEBUGGING ID MISMATCH ---")

# 1. Check GT Folder
reviewed_ids = set()
print(f"\nScanning {GT_DIR}...")
if not GT_DIR.exists():
    print("GT Dir does not exist!")
else:
    for f in GT_DIR.glob("gt_*.json"):
        print(f"  Found file: {f.name}")
        fname = f.name
        if len(fname) > 8:
            extracted_id = fname[3:-5]
            reviewed_ids.add(extracted_id)
            print(f"    -> Extracted ID: '{extracted_id}'")

# 2. Check Results File
print(f"\nScanning {RESULTS_PATH}...")
if not RESULTS_PATH.exists():
    print("Results file not found!")
else:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    items = data.get("results", [])
    print(f"  Total results: {len(items)}")
    
    # Check first 5 items
    for i, item in enumerate(items[:5]):
        raw_filename = item.get("filename", "unknown")
        safe_id_check = str(raw_filename).replace("\\", "_").replace("/", "_")
        
        status = "MATCHED (Would be hidden)" if safe_id_check in reviewed_ids else "MISSING (Still visible)"
        print(f"  Item {i}:")
        print(f"    Raw Filename: '{raw_filename}'")
        print(f"    Safe ID Check: '{safe_id_check}'")
        print(f"    Status: {status}")

print("\n---------------------------")
