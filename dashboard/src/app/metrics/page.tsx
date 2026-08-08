"use client";

import { useState, useEffect } from "react";
import {
  BarChart3,
  Clock,
  CheckCircle,
  XCircle,
  TrendingUp,
  ShieldAlert,
  RefreshCw,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn, formatConfidence } from "@/lib/utils";
import type { Job } from "@/types/api";

interface MetricsSummary {
  totalProcessed: number;
  succeeded: number;
  failed: number;
  autoClearRate: number;
  avgSpoofScore: number;
  avgConfidence: number;
  docTypeDistribution: Record<string, number>;
  riskDistribution: Record<string, number>;
  decisionDistribution: Record<string, number>;
}

function computeMetrics(jobs: Job[]): MetricsSummary {
  const completed = jobs.filter((j) => j.status === "SUCCEEDED" && j.result);
  const autoClear = completed.filter((j) => j.result?.status === "AUTO_CLEARED" || j.result?.status === "VALID");
  const withForensics = completed.filter((j) => j.result?.forensics);
  const withCalibration = completed.filter((j) => j.result?.calibration);

  const docTypes: Record<string, number> = {};
  const risks: Record<string, number> = {};
  const decisions: Record<string, number> = {};

  for (const j of completed) {
    const dt = j.result?.document_type ?? "unknown";
    docTypes[dt] = (docTypes[dt] || 0) + 1;
    const risk = j.result?.forensics?.risk_level ?? "N/A";
    risks[risk] = (risks[risk] || 0) + 1;
    const dec = j.result?.status ?? "UNKNOWN";
    decisions[dec] = (decisions[dec] || 0) + 1;
  }

  return {
    totalProcessed: jobs.length,
    succeeded: completed.length,
    failed: jobs.filter((j) => j.status === "FAILED").length,
    autoClearRate: completed.length > 0 ? autoClear.length / completed.length : 0,
    avgSpoofScore: withForensics.length > 0 ? withForensics.reduce((s, j) => s + (j.result?.forensics?.spoof_score ?? 0), 0) / withForensics.length : 0,
    avgConfidence: withCalibration.length > 0 ? withCalibration.reduce((s, j) => s + (j.result?.calibration?.calibrated_confidence ?? 0), 0) / withCalibration.length : 0,
    docTypeDistribution: docTypes,
    riskDistribution: risks,
    decisionDistribution: decisions,
  };
}

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({ opacity: 1, y: 0, transition: { delay: i * 0.06, duration: 0.4 } }),
};

function AnimatedBar({ label, value, max, color, colorClass }: { label: string; value: number; max: number; color?: string; colorClass?: string }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <span className={cn("font-medium capitalize", colorClass || "text-zinc-300")}>{label.replace(/_/g, " ")}</span>
        <span className="font-mono text-zinc-500">{value}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-zinc-800/80">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color || "#3b82f6" }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);

  const load = () => {
    try {
      const stored = JSON.parse(localStorage.getItem("kyc_jobs") || "[]") as Job[];
      setMetrics(computeMetrics(stored));
    } catch {
      setMetrics(computeMetrics([]));
    }
  };

  useEffect(() => { load(); }, []);

  if (!metrics) return null;

  const topCards = [
    { label: "Total Processed", value: metrics.totalProcessed, icon: BarChart3, color: "text-blue-400", bg: "bg-blue-500/10", ring: "ring-blue-500/20" },
    { label: "Succeeded", value: metrics.succeeded, icon: CheckCircle, color: "text-emerald-400", bg: "bg-emerald-500/10", ring: "ring-emerald-500/20" },
    { label: "Failed", value: metrics.failed, icon: XCircle, color: "text-red-400", bg: "bg-red-500/10", ring: "ring-red-500/20" },
    { label: "Auto-Clear Rate", value: formatConfidence(metrics.autoClearRate), icon: TrendingUp, color: "text-emerald-400", bg: "bg-emerald-500/10", ring: "ring-emerald-500/20" },
    { label: "Avg Confidence", value: metrics.avgConfidence > 0 ? formatConfidence(metrics.avgConfidence) : "---", icon: Clock, color: "text-amber-400", bg: "bg-amber-500/10", ring: "ring-amber-500/20" },
    { label: "Avg Spoof Score", value: metrics.avgSpoofScore > 0 ? formatConfidence(metrics.avgSpoofScore) : "---", icon: ShieldAlert, color: metrics.avgSpoofScore > 0.4 ? "text-red-400" : "text-emerald-400", bg: metrics.avgSpoofScore > 0.4 ? "bg-red-500/10" : "bg-emerald-500/10", ring: metrics.avgSpoofScore > 0.4 ? "ring-red-500/20" : "ring-emerald-500/20" },
  ];

  const riskColorMap: Record<string, string> = { LOW: "#10b981", MEDIUM: "#f59e0b", HIGH: "#ef4444", CRITICAL: "#dc2626" };
  const riskTextMap: Record<string, string> = { LOW: "text-emerald-400", MEDIUM: "text-amber-400", HIGH: "text-red-400", CRITICAL: "text-red-600" };

  return (
    <div className="space-y-6">
      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={0} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Metrics</h1>
          <p className="mt-1.5 text-sm text-zinc-500">Pipeline performance and processing analytics</p>
        </div>
        <button onClick={load} className="flex items-center gap-2 rounded-xl glass glass-hover px-4 py-2.5 text-sm font-medium text-zinc-300">
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </motion.div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {topCards.map((card, i) => {
          const Icon = card.icon;
          return (
            <motion.div key={card.label} initial="hidden" animate="visible" variants={fadeUp} custom={i + 1} className="glass glass-hover rounded-xl p-5">
              <div className="flex items-center justify-between">
                <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">{card.label}</p>
                <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg ring-1", card.bg, card.ring)}>
                  <Icon className={cn("h-4 w-4", card.color)} />
                </div>
              </div>
              <p className="mt-3 text-3xl font-bold tracking-tight text-white">{typeof card.value === "number" ? card.value : card.value}</p>
            </motion.div>
          );
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {[
          { title: "Document Types", data: metrics.docTypeDistribution },
          { title: "Risk Distribution", data: metrics.riskDistribution, isRisk: true },
          { title: "Decision Distribution", data: metrics.decisionDistribution },
        ].map((chart, i) => (
          <motion.div key={chart.title} initial="hidden" animate="visible" variants={fadeUp} custom={i + 7} className="glass rounded-xl p-5">
            <h3 className="mb-4 text-xs font-semibold text-zinc-400 uppercase tracking-wider">{chart.title}</h3>
            {Object.keys(chart.data).length > 0 ? (
              <div className="space-y-3">
                {Object.entries(chart.data).sort(([, a], [, b]) => b - a).map(([key, val]) => (
                  <AnimatedBar
                    key={key}
                    label={key}
                    value={val}
                    max={Math.max(...Object.values(chart.data), 1)}
                    color={chart.isRisk ? riskColorMap[key] : undefined}
                    colorClass={chart.isRisk ? riskTextMap[key] : undefined}
                  />
                ))}
              </div>
            ) : (
              <p className="text-xs text-zinc-700">No data yet</p>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
