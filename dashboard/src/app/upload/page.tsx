"use client";

import { useState, useCallback } from "react";
import { UploadZone } from "@/components/upload-zone";
import { JobCard } from "@/components/job-card";
import { ResultDetail } from "@/components/result-detail";
import { DecisionTimeline } from "@/components/decision-timeline";
import { useJobs } from "@/hooks/use-jobs";
import type { Job } from "@/types/api";

export default function UploadPage() {
  const { jobs, upload, pollUntilDone, loading } = useJobs();
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);

  const handleUpload = useCallback(
    async (file: File) => {
      const jobId = await upload(file);
      pollUntilDone(jobId);
      return jobId;
    },
    [upload, pollUntilDone]
  );

  const handleSelect = useCallback(
    (jobId: string) => {
      const job = jobs.find((j) => j.job_id === jobId);
      if (job) setSelectedJob(job);
    },
    [jobs]
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Upload Documents</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Submit document images for KYC verification. Supports batch upload.
        </p>
      </div>

      <UploadZone onUpload={handleUpload} disabled={loading} />

      {jobs.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-zinc-400">
            Processing Queue ({jobs.length})
          </h2>
          {jobs.map((job) => (
            <div key={job.job_id} className="flex gap-4">
              <div className="flex-1">
                <JobCard
                  job={job}
                  onPoll={pollUntilDone}
                  onSelect={handleSelect}
                />
              </div>
              {job.status === "SUCCEEDED" && job.result && (
                <div className="hidden w-64 xl:block">
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                    <p className="mb-3 text-xs font-semibold text-zinc-500">Decision Timeline</p>
                    <DecisionTimeline result={job.result} />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {selectedJob && (
        <ResultDetail job={selectedJob} onClose={() => setSelectedJob(null)} />
      )}
    </div>
  );
}
