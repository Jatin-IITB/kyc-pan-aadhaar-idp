"use client";

import { useState, useEffect, useMemo } from "react";
import { Search, Filter, RefreshCw } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { getJob } from "@/lib/api";
import { cn } from "@/lib/utils";
import { StatusBadge } from "@/components/status-badge";
import { ResultDetail } from "@/components/result-detail";
import { formatConfidence, riskColor, pipelineStatusLabel } from "@/lib/utils";
import type { Job, StatusFilter } from "@/types/api";

const FILTERS: StatusFilter[] = ["ALL", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED"];

export default function CasesPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [filter, setFilter] = useState<StatusFilter>("ALL");
  const [search, setSearch] = useState("");
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(false);

  const loadStoredJobs = () => {
    try {
      const stored = JSON.parse(localStorage.getItem("kyc_jobs") || "[]") as Job[];
      setJobs(stored);
    } catch {
      setJobs([]);
    }
  };

  useEffect(() => {
    loadStoredJobs();
  }, []);

  const refreshAll = async () => {
    setLoading(true);
    try {
      const stored = JSON.parse(localStorage.getItem("kyc_jobs") || "[]") as Job[];
      const updated = await Promise.all(
        stored.map(async (j) => {
          try { return await getJob(j.job_id); } catch { return j; }
        })
      );
      localStorage.setItem("kyc_jobs", JSON.stringify(updated));
      setJobs(updated);
    } finally {
      setLoading(false);
    }
  };

  const filtered = useMemo(() => {
    return jobs
      .filter((j) => filter === "ALL" || j.status === filter)
      .filter(
        (j) =>
          !search ||
          j.job_id.toLowerCase().includes(search.toLowerCase()) ||
          j.filename.toLowerCase().includes(search.toLowerCase()) ||
          j.result?.document_type?.toLowerCase().includes(search.toLowerCase())
      )
      .sort((a, b) => {
        const order = { RUNNING: 0, STARTED: 0, QUEUED: 1, SUCCEEDED: 2, FAILED: 3 };
        return (order[a.status] ?? 4) - (order[b.status] ?? 4);
      });
  }, [jobs, filter, search]);

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Cases</h1>
          <p className="mt-1.5 text-sm text-zinc-500">
            {jobs.length} total &middot; {jobs.filter((j) => j.status === "SUCCEEDED").length} completed
          </p>
        </div>
        <button
          onClick={refreshAll}
          disabled={loading}
          className="flex items-center gap-2 rounded-xl glass glass-hover px-4 py-2.5 text-sm font-medium text-zinc-300 transition-all disabled:opacity-50"
        >
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
          Refresh
        </button>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="flex flex-col gap-3 sm:flex-row sm:items-center"
      >
        <div className="relative flex-1">
          <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-600" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by job ID, filename, or document type..."
            className="w-full rounded-xl glass py-2.5 pl-10 pr-4 text-sm text-zinc-200 placeholder-zinc-600 outline-none focus:ring-1 focus:ring-blue-500/30 transition-all"
          />
        </div>
        <div className="flex items-center gap-1 rounded-xl glass p-1">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={cn(
                "relative rounded-lg px-3 py-1.5 text-xs font-medium transition-all duration-200",
                filter === f
                  ? "text-blue-400"
                  : "text-zinc-600 hover:text-zinc-300"
              )}
            >
              {filter === f && (
                <motion.div
                  layoutId="filter-bg"
                  className="absolute inset-0 rounded-lg bg-blue-500/10"
                  transition={{ duration: 0.2 }}
                />
              )}
              <span className="relative">{f}</span>
            </button>
          ))}
        </div>
      </motion.div>

      <AnimatePresence mode="wait">
        {filtered.length === 0 ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass rounded-2xl p-16 text-center"
          >
            <p className="text-zinc-600">
              {jobs.length === 0
                ? "No cases yet. Upload documents to get started."
                : "No cases match the current filter."}
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="table"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="overflow-x-auto rounded-2xl glass"
          >
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800/50">
                  <th className="px-5 py-3.5 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Job ID</th>
                  <th className="px-5 py-3.5 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">File</th>
                  <th className="px-5 py-3.5 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Type</th>
                  <th className="px-5 py-3.5 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Status</th>
                  <th className="px-5 py-3.5 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Decision</th>
                  <th className="px-5 py-3.5 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Confidence</th>
                  <th className="px-5 py-3.5 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Spoof</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/30">
                {filtered.map((job, i) => {
                  const decision = job.result ? pipelineStatusLabel(job.result.status) : null;
                  return (
                    <motion.tr
                      key={job.job_id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.03 }}
                      onClick={() => setSelectedJob(job)}
                      className="cursor-pointer transition-colors hover:bg-zinc-800/20"
                    >
                      <td className="px-5 py-3.5 font-mono text-xs text-zinc-500">{job.job_id.slice(0, 8)}...</td>
                      <td className="max-w-[200px] truncate px-5 py-3.5 font-medium text-zinc-200">{job.filename}</td>
                      <td className="px-5 py-3.5">
                        <span className="rounded-md bg-zinc-800/60 px-2 py-0.5 text-[11px] font-mono text-zinc-400 uppercase">{job.result?.document_type ?? "---"}</span>
                      </td>
                      <td className="px-5 py-3.5"><StatusBadge status={job.status} /></td>
                      <td className="px-5 py-3.5">
                        {decision ? (
                          <span className={cn("text-xs font-semibold", decision.color)}>{decision.label}</span>
                        ) : (
                          <span className="text-xs text-zinc-700">---</span>
                        )}
                      </td>
                      <td className="px-5 py-3.5 font-mono text-xs">
                        {job.result?.calibration ? (
                          <span className="text-zinc-300">{formatConfidence(job.result.calibration.calibrated_confidence)}</span>
                        ) : (
                          <span className="text-zinc-700">---</span>
                        )}
                      </td>
                      <td className="px-5 py-3.5 font-mono text-xs">
                        {job.result?.forensics ? (
                          <span className={riskColor(job.result.forensics.risk_level)}>
                            {(job.result.forensics.spoof_score * 100).toFixed(0)}%
                          </span>
                        ) : (
                          <span className="text-zinc-700">---</span>
                        )}
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </motion.div>
        )}
      </AnimatePresence>

      {selectedJob && (
        <ResultDetail job={selectedJob} onClose={() => setSelectedJob(null)} />
      )}
    </div>
  );
}
