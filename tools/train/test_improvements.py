# tools/train/test_improvements.py
import json
import sys
import re
from pathlib import Path

# Fix import path
sys.path.append(str(Path(__file__).parent.parent.parent))

DATASET_PATH = Path("data/training/dataset_v1.jsonl")

def validate_aadhaar_simple(data):
    """Simple validator to check against Gold Data rules"""
    errors = []
    
    # Gender Logic (Strict)
    gender = data.get("gender", "")
    allowed = {"male", "female", "other"}
    if gender and gender.lower() not in allowed:
        # Logic Improvement Opportunity: If this fails, we need better normalization!
        errors.append(f"Gender '{gender}' is not in allowed list")
        
    # Date Logic (DD/MM/YYYY)
    dob = data.get("date_of_birth", "")
    if dob:
        if not re.match(r"^(\d{4}|\d{2}/\d{2}/\d{4})$", dob):
             # Logic Improvement Opportunity: Support YYYY or other formats?
             errors.append(f"DOB '{dob}' does not match DD/MM/YYYY")
             
    return errors

def main():
    if not DATASET_PATH.exists():
        print("Run prepare_data.py first!")
        return

    print(f"🧪 Testing Logic against {DATASET_PATH}...")
    
    stats = {"total": 0, "passed": 0, "failed": 0}
    failures = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stats["total"] += 1
            record = json.loads(line)
            gt = record["ground_truth"]
            
            # Run Validation
            if record["document_type"] == "aadhaar":
                errs = validate_aadhaar_simple(gt)
            else:
                continue
                # errs = [] # Skip PAN for now as you only have Aadhaar

            if not errs:
                stats["passed"] += 1
            else:
                stats["failed"] += 1
                failures.append({
                    "id": record["filename"],
                    "errors": errs,
                    "data": gt
                })

    print(f"\n📊 Results: {stats['passed']}/{stats['total']} Gold Records passed validation.")
    
    if failures:
        print("\n❌ FAILURES (Action Items for Logic Improvements):")
        for f in failures[:5]: # Show top 5
            print(f"  ID: {f['id']}")
            for e in f['errors']:
                print(f"    - {e}")
            print(f"    - Raw Data: {f['data']}")
            print("")
            
    if stats['failed'] == 0:
        print("\n✅ All Gold Data is valid! You can deploy this dataset for training.")

if __name__ == "__main__":
    main()
