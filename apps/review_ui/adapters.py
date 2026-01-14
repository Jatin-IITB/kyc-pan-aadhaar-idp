# apps/review_ui/adapters.py
import json
import os
import tempfile
from pathlib import Path
from typing import List, Set
from apps.review_ui.domain import ReviewJob, ReviewResult

class FileSystemAdapter:
    def __init__(self, eval_results_path: str, eval_images_roots: List[str]):
        self.results_path = Path(eval_results_path)
        self.image_roots = [Path(p) for p in eval_images_roots]
        self.gt_dir = Path("data/reviewed_ground_truth")
        self.gt_dir.mkdir(parents=True, exist_ok=True)

    def _get_reviewed_ids(self) -> Set[str]:
        reviewed_ids = set()
        for f in self.gt_dir.glob("gt_*.json"):
            fname = f.name
            if len(fname) > 8:
                safe_id = fname[3:-5]
                reviewed_ids.add(safe_id)
        return reviewed_ids
        
    def _find_image(self, filename: str) -> str:
        if not filename: return ""
        for root in self.image_roots:
            if not root.exists(): continue
            found = list(root.rglob(filename))
            if found:
                return str(found[0])
        return ""

    def get_jobs(self, status_filter: str = "ALL") -> List[ReviewJob]:
        if not self.results_path.exists():
            return []
            
        try:
            data = json.loads(self.results_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        reviewed_ids = self._get_reviewed_ids()
        jobs = []
        items = data.get("results", []) if isinstance(data, dict) else data
        
        for item in items:
            fname = item.get("filename", "unknown")
            safe_id_check = str(fname).replace("\\", "_").replace("/", "_")

            # if safe_id_check in reviewed_ids:
                # continue

            # --- STATUS LOGIC START ---
            ok = item.get("ok", False)
            res = item.get("result", {}) or {}
            
            pipeline_status = res.get("status", "UNKNOWN")
            quality = res.get("quality_check", {})
            validation = res.get("validation", {})
            
            # Map Pipeline Status to UI Status
            if pipeline_status == "REJECTED_QUALITY":
                status = "REJECTED"
                error = quality.get("rejection_reason", "Unknown Quality Issue")
            elif pipeline_status == "REJECTED_CONTENT":
                status = "REJECTED"
                error = "Back side or Empty Content"
            else:
                # Normal validation logic
                is_valid = validation.get("is_valid", False)
                status = "VALID" if (ok and is_valid) else "INVALID"
            # --- STATUS LOGIC END ---

            if status_filter != "ALL" and status != status_filter:
                continue
                
            doc_type = res.get("document_type", "unknown")
            extraction = res.get("extraction", {})
            # Use specific error if we set one above, else fallback to validation message
            final_error = error if status == "REJECTED" else validation.get("message")
            
            jobs.append(ReviewJob(
                id=fname,
                source="EVAL",
                status=status,
                document_type=doc_type,
                image_path=self._find_image(fname),
                extraction=extraction,
                validation_error=final_error
            ))
            
        return jobs

    def save_review(self, review: ReviewResult) -> str:
        safe_id = str(review.job_id).replace("\\", "_").replace("/", "_")
        out_path = self.gt_dir / f"gt_{safe_id}.json"
        
        payload = {
            "job_id": review.job_id,
            "reviewer": review.reviewer,
            "timestamp": review.reviewed_at,
            "correction": review.corrected_data,
            "notes": review.notes
        }
        
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(self.gt_dir), encoding="utf-8") as tf:
            tf.write(json.dumps(payload, indent=2))
            tmpname = tf.name
        Path(tmpname).replace(out_path)
        
        return str(out_path)
