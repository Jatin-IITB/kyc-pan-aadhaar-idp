"use client";

import {
  Camera,
  Search,
  FileText,
  CheckCircle2,
  Shield,
  BookOpen,
  Users,
  Scale,
  Stamp,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { JobResult } from "@/types/api";

interface TimelineEvent {
  icon: typeof Camera;
  label: string;
  status: "pass" | "fail" | "warn" | "info";
  detail: string;
}

function buildTimeline(result: JobResult): TimelineEvent[] {
  const events: TimelineEvent[] = [];

  // Quality Gate
  const q = result.quality_check;
  events.push({
    icon: Camera,
    label: "Quality Gate",
    status: q.rejection_reason ? "fail" : "pass",
    detail: q.rejection_reason || `Blur: ${q.blur_score.toFixed(0)}, Exposure OK`,
  });

  // Classification
  events.push({
    icon: Search,
    label: "Classification",
    status: "info",
    detail: `Detected: ${result.document_type.toUpperCase()}`,
  });

  // Extraction
  const fieldCount = Object.keys(result.extraction).length;
  events.push({
    icon: FileText,
    label: "Extraction",
    status: fieldCount > 0 ? "pass" : "warn",
    detail: fieldCount > 0 ? `${fieldCount} fields extracted` : "No fields extracted",
  });

  // Validation
  events.push({
    icon: CheckCircle2,
    label: "Validation",
    status: result.validation.is_valid ? "pass" : "fail",
    detail: result.validation.message,
  });

  // Forensics
  if (result.forensics) {
    events.push({
      icon: Shield,
      label: "Forensics",
      status:
        result.forensics.risk_level === "LOW"
          ? "pass"
          : result.forensics.risk_level === "CRITICAL"
            ? "fail"
            : "warn",
      detail: `Spoof: ${(result.forensics.spoof_score * 100).toFixed(1)}% — ${result.forensics.risk_level}`,
    });
  }

  // Policy
  if (result.policy) {
    events.push({
      icon: BookOpen,
      label: "Policy Compliance",
      status: result.policy.overall_status === "COMPLIANT" ? "pass" : result.policy.overall_status === "NON_COMPLIANT" ? "fail" : "warn",
      detail: result.policy.overall_status,
    });
  }

  // Cross-Doc
  if (result.cross_doc) {
    events.push({
      icon: Users,
      label: "Cross-Document",
      status: result.cross_doc.contradictions.length === 0 ? "pass" : "warn",
      detail: `Consistency: ${(result.cross_doc.consistency_score * 100).toFixed(0)}%`,
    });
  }

  // Calibration
  if (result.calibration) {
    events.push({
      icon: Scale,
      label: "Calibration",
      status: "info",
      detail: `Confidence: ${(result.calibration.calibrated_confidence * 100).toFixed(1)}%`,
    });
  }

  // Final Decision
  events.push({
    icon: result.status.startsWith("REJECTED") ? XCircle : Stamp,
    label: "Decision",
    status: result.status === "AUTO_CLEARED" || result.status === "VALID" ? "pass" : result.status.startsWith("REVIEW") ? "warn" : "fail",
    detail: result.status.replace(/_/g, " "),
  });

  return events;
}

const statusStyles = {
  pass: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
  fail: "border-red-500/30 bg-red-500/10 text-red-400",
  warn: "border-amber-500/30 bg-amber-500/10 text-amber-400",
  info: "border-blue-500/30 bg-blue-500/10 text-blue-400",
};

const lineColor = {
  pass: "bg-emerald-500/40",
  fail: "bg-red-500/40",
  warn: "bg-amber-500/40",
  info: "bg-blue-500/40",
};

export function DecisionTimeline({ result }: { result: JobResult }) {
  const events = buildTimeline(result);

  return (
    <div className="space-y-0">
      {events.map((evt, i) => {
        const Icon = evt.icon;
        const isLast = i === events.length - 1;
        return (
          <div key={i} className="flex gap-4">
            <div className="flex flex-col items-center">
              <div className={cn("flex h-8 w-8 items-center justify-center rounded-full border", statusStyles[evt.status])}>
                <Icon className="h-4 w-4" />
              </div>
              {!isLast && <div className={cn("w-0.5 flex-1 min-h-[24px]", lineColor[evt.status])} />}
            </div>
            <div className="pb-6">
              <p className="text-sm font-medium text-zinc-200">{evt.label}</p>
              <p className="text-xs text-zinc-500">{evt.detail}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
