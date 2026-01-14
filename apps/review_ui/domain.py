# apps/review_ui/domain.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class ReviewJob:
    id: str
    source: str          # "LIVE" or "EVAL"
    status: str          # "VALID" or "INVALID"
    document_type: str
    image_path: str      # Local path or Presigned URL
    extraction: Dict[str, Any]
    validation_error: Optional[str]
    
@dataclass
class ReviewResult:
    job_id: str
    reviewer: str
    corrected_data: Dict[str, Any]
    reviewed_at: str
    notes: str
