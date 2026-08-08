import type { Job, AuditEvent } from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...init?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export async function submitJob(file: File, docType = "auto"): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({ doc_type: docType });
  return request(`/jobs?${params}`, { method: "POST", body: form });
}

export async function getJob(jobId: string): Promise<Job> {
  return request(`/jobs/${jobId}`);
}

export async function getHealth(): Promise<{ status: string }> {
  return request("/health");
}

export async function getMetrics(): Promise<string> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Metrics ${res.status}: ${await res.text()}`);
  return res.text();
}

export async function getCaseAudit(caseId: string): Promise<{ events: AuditEvent[] }> {
  return request(`/v1/cases/${caseId}/audit`);
}

export function subscribeToCaseProgress(
  caseId: string,
  onEvent: (data: Record<string, unknown>) => void,
  onError?: (err: Event) => void
): () => void {
  const es = new EventSource(`${API_BASE}/v1/cases/${caseId}/progress`);
  es.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch {
      // ignore parse errors
    }
  };
  if (onError) es.onerror = onError;
  return () => es.close();
}
