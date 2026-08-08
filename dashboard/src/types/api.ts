export interface QualityCheck {
  blur_score: number;
  overexposed_ratio: number;
  underexposed_ratio: number;
  is_blurry: boolean;
  is_overexposed: boolean;
  is_underexposed: boolean;
  rejection_reason: string | null;
}

export interface FieldData {
  value: string;
  det_conf: number;
  ocr_conf: number;
  bbox: [number, number, number, number] | null;
}

export interface ForensicsResult {
  spoof_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  recommendation: string;
  component_scores: Record<string, number>;
  evidence: Array<{ type: string; score: number; detail: string }>;
}

export interface CalibrationResult {
  calibrated_confidence: number;
  recommendation: string;
  raw_scores: Record<string, number>;
  overrides: Array<{ rule: string; action: string; reason: string }>;
}

export interface PolicyCheck {
  status: "PASS" | "FAIL" | "NOT_APPLICABLE";
  requirement: string;
  explanation: string;
}

export interface PolicyResult {
  overall_status: "COMPLIANT" | "NON_COMPLIANT" | "PARTIAL";
  checks: PolicyCheck[];
}

export interface CrossDocResult {
  consistency_score: number;
  recommendation: string;
  contradictions: Array<{
    field: string;
    severity: "CRITICAL" | "WARNING" | "INFO";
    values: string[];
    docs: string[];
  }>;
  entity_resolution: {
    match_score: number;
    canonical_name: string;
    is_same_person: boolean;
  };
}

export interface JobResult {
  document_type: string;
  quality_check: QualityCheck;
  extraction: Record<string, FieldData | string>;
  validation: { is_valid: boolean; message: string };
  status: string;
  forensics?: ForensicsResult;
  calibration?: CalibrationResult;
  policy?: PolicyResult;
  cross_doc?: CrossDocResult;
}

export interface Job {
  job_id: string;
  status: "QUEUED" | "STARTED" | "RUNNING" | "SUCCEEDED" | "FAILED";
  filename: string;
  ok?: boolean;
  error?: string;
  result?: JobResult;
}

export interface CaseResponse {
  case_id: string;
  status: string;
  documents: Array<{
    job_id: string;
    filename: string;
    doc_type: string;
    status: string;
  }>;
  created_at: string;
}

export interface AuditEvent {
  event_type: string;
  timestamp: string;
  payload: Record<string, unknown>;
  hash: string;
  prev_hash: string;
}

export type StatusFilter = "ALL" | "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED";
