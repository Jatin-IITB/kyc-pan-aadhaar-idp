import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatConfidence(val: number): string {
  return `${(val * 100).toFixed(1)}%`;
}

export function riskColor(level: string): string {
  switch (level) {
    case "LOW": return "text-emerald-400";
    case "MEDIUM": return "text-amber-400";
    case "HIGH": return "text-red-400";
    case "CRITICAL": return "text-red-600";
    default: return "text-zinc-400";
  }
}

export function statusColor(status: string): string {
  switch (status) {
    case "SUCCEEDED": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
    case "FAILED": return "bg-red-500/20 text-red-400 border-red-500/30";
    case "RUNNING":
    case "STARTED": return "bg-blue-500/20 text-blue-400 border-blue-500/30";
    case "QUEUED": return "bg-zinc-500/20 text-zinc-400 border-zinc-500/30";
    default: return "bg-zinc-500/20 text-zinc-400 border-zinc-500/30";
  }
}

export function pipelineStatusLabel(status: string): { label: string; color: string } {
  switch (status) {
    case "VALID": return { label: "Valid", color: "text-emerald-400" };
    case "REJECTED_QUALITY": return { label: "Rejected (Quality)", color: "text-red-400" };
    case "REJECTED_CONTENT": return { label: "Rejected (Content)", color: "text-red-400" };
    case "REJECTED_SPOOF": return { label: "Rejected (Spoof)", color: "text-red-500" };
    case "REJECTED_CALIBRATION": return { label: "Rejected (Calibration)", color: "text-red-400" };
    case "REVIEW": return { label: "Needs Review", color: "text-amber-400" };
    case "REVIEW_SPOOF": return { label: "Review (Spoof)", color: "text-amber-500" };
    case "AUTO_CLEARED": return { label: "Auto-Cleared", color: "text-emerald-400" };
    default: return { label: status, color: "text-zinc-400" };
  }
}
