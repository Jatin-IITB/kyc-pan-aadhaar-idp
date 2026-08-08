"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { getJob, submitJob } from "@/lib/api";
import type { Job } from "@/types/api";

const STORAGE_KEY = "kyc_jobs";
const MAX_POLL_ATTEMPTS = 60;
const POLL_INTERVAL = 2000;

function loadFromStorage(): Job[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveToStorage(jobs: Job[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs));
}

export function useJobs() {
  const [jobs, setJobs] = useState<Job[]>(loadFromStorage);
  const [loading, setLoading] = useState(false);
  const abortControllers = useRef(new Map<string, AbortController>());

  useEffect(() => {
    saveToStorage(jobs);
  }, [jobs]);

  useEffect(() => {
    return () => {
      for (const controller of abortControllers.current.values()) {
        controller.abort();
      }
    };
  }, []);

  const upload = useCallback(async (file: File) => {
    setLoading(true);
    try {
      const { job_id } = await submitJob(file);
      const job: Job = { job_id, status: "QUEUED", filename: file.name };
      setJobs((prev) => [job, ...prev]);
      return job_id;
    } finally {
      setLoading(false);
    }
  }, []);

  const refresh = useCallback(async (jobId: string) => {
    try {
      const updated = await getJob(jobId);
      setJobs((prev) => prev.map((j) => (j.job_id === jobId ? { ...updated, filename: j.filename || updated.filename } : j)));
      return updated;
    } catch {
      return null;
    }
  }, []);

  const pollUntilDone = useCallback(
    (jobId: string) => {
      const existing = abortControllers.current.get(jobId);
      if (existing) existing.abort();

      const controller = new AbortController();
      abortControllers.current.set(jobId, controller);

      let attempts = 0;
      const poll = () => {
        if (controller.signal.aborted || attempts >= MAX_POLL_ATTEMPTS) {
          abortControllers.current.delete(jobId);
          return;
        }
        attempts++;
        refresh(jobId).then((updated) => {
          if (controller.signal.aborted) return;
          if (!updated || updated.status === "SUCCEEDED" || updated.status === "FAILED") {
            abortControllers.current.delete(jobId);
            return;
          }
          setTimeout(poll, POLL_INTERVAL);
        });
      };
      poll();
    },
    [refresh]
  );

  const remove = useCallback((jobId: string) => {
    const controller = abortControllers.current.get(jobId);
    if (controller) {
      controller.abort();
      abortControllers.current.delete(jobId);
    }
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
  }, []);

  return { jobs, upload, refresh, pollUntilDone, remove, loading };
}
