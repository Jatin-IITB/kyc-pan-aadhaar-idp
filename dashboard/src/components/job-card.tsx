"use client";

import { useEffect, useRef } from "react";
import { FileText, Clock, CheckCircle2, XCircle, Loader2, ChevronRight } from "lucide-react";
import { motion } from "framer-motion";
import { cn, formatConfidence } from "@/lib/utils";
import { StatusBadge, PipelineStatusBadge } from "./status-badge";
import type { Job } from "@/types/api";

interface JobCardProps {
  job: Job;
  onPoll: (jobId: string) => void;
  onSelect: (jobId: string) => void;
}

export function JobCard({ job, onPoll, onSelect }: JobCardProps) {
  const pollingRef = useRef(false);

  useEffect(() => {
    if (
      (job.status === "QUEUED" || job.status === "STARTED" || job.status === "RUNNING") &&
      !pollingRef.current
    ) {
      pollingRef.current = true;
      onPoll(job.job_id);
    }
  }, [job.status, job.job_id, onPoll]);

  const result = job.result;
  const isProcessing = job.status === "RUNNING" || job.status === "STARTED";
  const Icon = job.status === "SUCCEEDED" ? CheckCircle2 : job.status === "FAILED" ? XCircle : isProcessing ? Loader2 : Clock;
  const iconColor = job.status === "SUCCEEDED" ? "text-emerald-400" : job.status === "FAILED" ? "text-red-400" : "text-blue-400";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      onClick={() => job.status === "SUCCEEDED" && onSelect(job.job_id)}
      className={cn(
        "group flex items-center gap-4 glass glass-hover rounded-xl p-4",
        job.status === "SUCCEEDED" && "cursor-pointer"
      )}
    >
      <div className={cn("flex-shrink-0", iconColor)}>
        <Icon className={cn("h-5 w-5", isProcessing && "animate-spin")} />
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5 text-zinc-600" />
          <span className="truncate text-sm font-medium text-zinc-200">{job.filename}</span>
        </div>
        <div className="mt-1.5 flex items-center gap-2.5">
          <StatusBadge status={job.status} />
          {result && <PipelineStatusBadge status={result.status} />}
          {result?.document_type && (
            <span className="text-[11px] font-mono text-zinc-600 uppercase">{result.document_type}</span>
          )}
        </div>
      </div>

      {isProcessing && (
        <div className="hidden sm:block">
          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-zinc-800">
            <div className="h-full w-full animate-pulse rounded-full bg-blue-500/50 shimmer relative" />
          </div>
          <p className="mt-1 text-[10px] text-zinc-600 text-right">Processing...</p>
        </div>
      )}

      {result?.forensics && (
        <div className="hidden flex-shrink-0 text-right sm:block">
          <p className="text-[10px] font-medium text-zinc-600 uppercase">Spoof</p>
          <p className={cn("text-sm font-mono font-bold", result.forensics.spoof_score > 0.5 ? "text-red-400" : result.forensics.spoof_score > 0.3 ? "text-amber-400" : "text-emerald-400")}>
            {formatConfidence(result.forensics.spoof_score)}
          </p>
        </div>
      )}

      {result?.calibration && (
        <div className="hidden flex-shrink-0 text-right sm:block">
          <p className="text-[10px] font-medium text-zinc-600 uppercase">Confidence</p>
          <p className="text-sm font-mono font-bold text-blue-400">
            {formatConfidence(result.calibration.calibrated_confidence)}
          </p>
        </div>
      )}

      {job.status === "SUCCEEDED" && (
        <ChevronRight className="h-4 w-4 flex-shrink-0 text-zinc-700 transition-all group-hover:text-zinc-400 group-hover:translate-x-0.5" />
      )}
    </motion.div>
  );
}
