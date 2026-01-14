# tools/train/validate_improvements.py
import json
import cv2
import sys
import re
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from apps.workers.pipeline_loader import get_pipeline

console = Console()

# --- CONSTANTS ---
ACCEPTED_STATUSES = {"SUCCESS", "PARTIAL_SUCCESS"}

def normalize_val(v):
    """Normalize for comparison (strip case/spaces/punctuation)"""
    if v is None: return ""
    # Remove all non-alphanumeric chars for strict comparison
    s = str(v).lower().strip()
    return re.sub(r'[\W_]+', '', s)

def main():
    DATASET_PATH = Path("data/training/dataset_v1.jsonl")

    if not DATASET_PATH.exists():
        console.print(f"[red]Dataset not found: {DATASET_PATH}.[/red]")
        return

    console.print("[bold blue]🚀 Loading Pipeline...[/bold blue]")
    pipeline = get_pipeline()

    console.print(f"[bold blue]📂 Loading Dataset: {DATASET_PATH}[/bold blue]")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # Metrics
    total_docs = 0
    rejected_docs = 0
    accepted_docs = 0
    
    # Field Stats: field_name -> [correct_count, total_count]
    field_stats = defaultdict(lambda: [0, 0])
    
    failures = []

    for record in tqdm(records, desc="Evaluating"):
        img_path = record["image_path"]
        gt = record["ground_truth"]
        doc_type = record["document_type"]

        img = cv2.imread(img_path)
        if img is None: continue
        total_docs += 1

        try:
            # Run Pipeline
            result = pipeline.extract_from_bgr(img, doc_type)
            
            # Check Status
            status = result.get("status", "UNKNOWN")
            extraction = result.get("extraction", {})
            
            # --- STATUS CHECK ---
            if status not in ACCEPTED_STATUSES:
                rejected_docs += 1
                continue
            
            accepted_docs += 1

            # --- FIELD COMPARISON ---
            for field, gt_val in gt.items():
                if not gt_val: continue

                # Get predicted value
                pred_obj = extraction.get(field, {})
                pred_val = pred_obj.get("value", "") if isinstance(pred_obj, dict) else str(pred_obj)

                # Normalize
                norm_gt = normalize_val(gt_val)
                norm_pred = normalize_val(pred_val)

                # Update Stats
                field_stats[field][1] += 1 # Total count for this field

                if norm_gt == norm_pred:
                    field_stats[field][0] += 1 # Correct count
                else:
                    failures.append({
                        "file": Path(img_path).name,
                        "field": field,
                        "expected": gt_val,
                        "got": pred_val,
                        "status": status
                    })

        except Exception as e:
            console.print(f"[red]Error on {Path(img_path).name}: {e}[/red]")
            continue

    # --- REPORTING ---
    
    console.print("\n[bold]📊 EVALUATION SUMMARY[/bold]")
    
    # 1. Document Stats
    rejection_rate = (rejected_docs / total_docs * 100) if total_docs else 0
    console.print(f"Total Documents: {total_docs}")
    console.print(f"[cyan]Accepted:        {accepted_docs}[/cyan]")
    console.print(f"[yellow]Rejected:        {rejected_docs} ({rejection_rate:.1f}%)[/yellow]")

    # 2. Overall Accuracy
    total_fields = sum(s[1] for s in field_stats.values())
    correct_fields = sum(s[0] for s in field_stats.values())
    clean_accuracy = (correct_fields / total_fields * 100) if total_fields else 0
    
    console.print(f"\n[bold green]✅ Clean Field Accuracy: {clean_accuracy:.2f}% ({correct_fields}/{total_fields})[/bold green]")
    
    # 3. Field-Wise Breakdown Table
    console.print("\n[bold]🔍 Accuracy by Field Type:[/bold]")
    ftable = Table(show_header=True, header_style="bold blue")
    ftable.add_column("Field")
    ftable.add_column("Accuracy", justify="right")
    ftable.add_column("Counts", justify="right")
    
    for field, stats in sorted(field_stats.items()):
        acc = (stats[0] / stats[1] * 100) if stats[1] else 0
        color = "green" if acc > 85 else "yellow" if acc > 70 else "red"
        ftable.add_row(field, f"[{color}]{acc:.1f}%[/{color}]", f"{stats[0]}/{stats[1]}")
    console.print(ftable)

    # 4. Failures
    if failures:
        console.print("\n[bold red]❌ Top Failures (Accepted Docs Only):[/bold red]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("File")
        table.add_column("Field")
        table.add_column("Expected")
        table.add_column("Got")
        
        for f in failures[:15]:
            table.add_row(f["file"], f["field"], str(f["expected"]), str(f["got"]))
        console.print(table)

if __name__ == "__main__":
    main()
