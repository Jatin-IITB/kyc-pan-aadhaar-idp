"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Shield,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  FileSearch,
  Scale,
  Users,
  BookOpen,
  X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn, formatConfidence, riskColor } from "@/lib/utils";
import { PipelineStatusBadge } from "./status-badge";
import type { Job, FieldData, ForensicsResult } from "@/types/api";

type Tab = "fields" | "forensics" | "calibration" | "crossdoc" | "policy";

const TABS: { key: Tab; label: string; icon: typeof Shield }[] = [
  { key: "fields", label: "Fields", icon: FileSearch },
  { key: "forensics", label: "Forensics", icon: Shield },
  { key: "calibration", label: "Calibration", icon: Scale },
  { key: "crossdoc", label: "Cross-Doc", icon: Users },
  { key: "policy", label: "Policy", icon: BookOpen },
];

const tabMotion = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.25 } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.15 } },
};

export function ResultDetail({ job, onClose }: { job: Job; onClose: () => void }) {
  const [tab, setTab] = useState<Tab>("fields");
  const result = job.result;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [handleKeyDown]);

  if (!result) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-4 backdrop-blur-md"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      role="dialog"
      aria-modal="true"
      aria-label={`Result detail for ${job.filename}`}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96 }}
        transition={{ duration: 0.25, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="my-8 w-full max-w-4xl rounded-2xl border border-zinc-800/50 bg-zinc-950 shadow-2xl shadow-black/50"
      >
        <div className="flex items-center justify-between border-b border-zinc-800/50 px-6 py-4">
          <div>
            <h2 className="text-lg font-bold tracking-tight text-white">{job.filename}</h2>
            <div className="mt-1.5 flex items-center gap-3">
              <span className="rounded-md bg-zinc-800 px-2 py-0.5 text-[11px] font-semibold text-zinc-400 uppercase tracking-wide">{result.document_type}</span>
              <PipelineStatusBadge status={result.status} />
              {result.calibration && (
                <span className="text-[11px] font-mono text-zinc-500">{formatConfidence(result.calibration.calibrated_confidence)} conf</span>
              )}
            </div>
          </div>
          <button onClick={onClose} className="rounded-xl p-2 text-zinc-500 hover:bg-zinc-800 hover:text-white transition-all">
            <X className="h-5 w-5" />
          </button>
        </div>

        {result.quality_check.rejection_reason && (
          <div className="mx-6 mt-4 flex items-center gap-2 rounded-xl bg-amber-500/10 border border-amber-500/20 px-4 py-2.5">
            <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
            <span className="text-sm text-amber-300">{result.quality_check.rejection_reason}</span>
          </div>
        )}

        <div className="flex gap-1 border-b border-zinc-800/50 px-6 pt-4">
          {TABS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={cn(
                "relative flex items-center gap-1.5 rounded-t-xl px-4 py-2.5 text-sm font-medium transition-all duration-200",
                tab === key
                  ? "text-blue-400"
                  : "text-zinc-600 hover:text-zinc-300"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
              {tab === key && (
                <motion.div
                  layoutId="tab-indicator"
                  className="absolute bottom-0 left-0 right-0 h-[2px] bg-blue-400 rounded-full"
                  transition={{ duration: 0.25 }}
                />
              )}
            </button>
          ))}
        </div>

        <div className="p-6 min-h-[300px]">
          <AnimatePresence mode="wait">
            <motion.div key={tab} {...tabMotion}>
              {tab === "fields" && <FieldsPanel extraction={result.extraction} />}
              {tab === "forensics" && <ForensicsPanel forensics={result.forensics} />}
              {tab === "calibration" && <CalibrationPanel calibration={result.calibration} status={result.status} />}
              {tab === "crossdoc" && <CrossDocPanel crossDoc={result.cross_doc} />}
              {tab === "policy" && <PolicyPanel policy={result.policy} />}
            </motion.div>
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}

function FieldsPanel({ extraction }: { extraction: Record<string, FieldData | string> }) {
  if (!extraction || Object.keys(extraction).length === 0) {
    return <Empty msg="No extraction data available." />;
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {Object.entries(extraction).map(([key, val], i) => {
        const isObj = typeof val === "object" && val !== null;
        const value = isObj ? (val as FieldData).value : String(val);
        const detConf = isObj ? (val as FieldData).det_conf : 0;
        const ocrConf = isObj ? (val as FieldData).ocr_conf : 0;
        return (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="glass rounded-xl p-4"
          >
            <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">{key.replace(/_/g, " ")}</p>
            <p className="mt-1.5 font-mono text-base font-medium text-white">{value || "---"}</p>
            {isObj && (
              <div className="mt-2.5 flex gap-4 text-[11px]">
                <span className="text-zinc-600">det <span className="font-mono text-zinc-500">{formatConfidence(detConf)}</span></span>
                <span className="text-zinc-600">ocr <span className="font-mono text-zinc-500">{formatConfidence(ocrConf)}</span></span>
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}

function ForensicsPanel({ forensics }: { forensics?: ForensicsResult }) {
  if (!forensics) return <Empty msg="No forensics data available." />;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-3">
        <GaugeCard label="Spoof Score" value={forensics.spoof_score} format={(v) => v.toFixed(3)} color={forensics.spoof_score > 0.5 ? "red" : forensics.spoof_score > 0.3 ? "amber" : "emerald"} />
        <div className="glass rounded-xl p-4 text-center">
          <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Risk Level</p>
          <p className={cn("mt-2 text-xl font-bold", riskColor(forensics.risk_level))}>{forensics.risk_level}</p>
        </div>
        <div className="glass rounded-xl p-4 text-center">
          <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Recommendation</p>
          <p className="mt-2 text-xl font-bold text-zinc-200">{forensics.recommendation}</p>
        </div>
      </div>

      {Object.keys(forensics.component_scores).length > 0 && (
        <div>
          <h4 className="mb-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider">Component Breakdown</h4>
          <div className="space-y-2.5">
            {Object.entries(forensics.component_scores).map(([k, v], i) => (
              <motion.div key={k} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
                <ProgressBar label={k.replace(/_/g, " ")} value={v} />
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {forensics.evidence.length > 0 && (
        <div>
          <h4 className="mb-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider">Evidence Flags</h4>
          <div className="space-y-2">
            {forensics.evidence.map((e, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="flex items-start gap-3 glass rounded-xl p-3.5"
              >
                {e.score > 0.5 ? <XCircle className="mt-0.5 h-4 w-4 text-red-400 shrink-0" /> : <AlertTriangle className="mt-0.5 h-4 w-4 text-amber-400 shrink-0" />}
                <div>
                  <p className="text-sm font-medium text-zinc-200">{e.type} <span className="font-mono text-zinc-500">({e.score.toFixed(2)})</span></p>
                  <p className="text-xs text-zinc-500 mt-0.5">{e.detail}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CalibrationPanel({ calibration, status }: { calibration?: NonNullable<Job["result"]>["calibration"]; status: string }) {
  if (!calibration) return <Empty msg="No calibration data available." />;
  const weights: Record<string, number> = { extraction: 0.35, forensics: 0.25, policy: 0.25, cross_doc: 0.15 };
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-3">
        <GaugeCard label="Calibrated Confidence" value={calibration.calibrated_confidence} format={(v) => formatConfidence(v)} color={calibration.calibrated_confidence > 0.9 ? "emerald" : calibration.calibrated_confidence > 0.7 ? "blue" : "amber"} />
        <div className="glass rounded-xl p-4 text-center">
          <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Recommendation</p>
          <p className="mt-2 text-xl font-bold text-zinc-200">{calibration.recommendation}</p>
        </div>
        <div className="glass rounded-xl p-4 text-center">
          <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Decision</p>
          <p className={cn("mt-2 text-xl font-bold", status === "AUTO_CLEARED" ? "text-emerald-400" : status.startsWith("REJECTED") ? "text-red-400" : "text-amber-400")}>{status.replace(/_/g, " ")}</p>
        </div>
      </div>

      <div>
        <h4 className="mb-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider">Signal Weights</h4>
        <div className="space-y-2.5">
          {Object.entries(calibration.raw_scores).map(([k, v], i) => (
            <motion.div key={k} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
              <ProgressBar
                label={`${k.replace(/_/g, " ")} (${((weights[k] ?? 0) * 100).toFixed(0)}%)`}
                value={v}
              />
            </motion.div>
          ))}
        </div>
      </div>

      {calibration.overrides.length > 0 && (
        <div>
          <h4 className="mb-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider">Override Rules</h4>
          {calibration.overrides.map((o, i) => (
            <div key={i} className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3.5">
              <p className="text-sm font-medium text-amber-300">{o.rule} &#8594; {o.action}</p>
              <p className="text-xs text-zinc-500 mt-0.5">{o.reason}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CrossDocPanel({ crossDoc }: { crossDoc?: NonNullable<Job["result"]>["cross_doc"] }) {
  if (!crossDoc) return <Empty msg="No cross-document analysis available (single document)." />;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3">
        <GaugeCard label="Consistency Score" value={crossDoc.consistency_score} format={formatConfidence} color={crossDoc.consistency_score > 0.8 ? "emerald" : "amber"} />
        <div className="glass rounded-xl p-4 text-center">
          <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Recommendation</p>
          <p className="mt-2 text-xl font-bold text-zinc-200">{crossDoc.recommendation}</p>
        </div>
      </div>
      {crossDoc.contradictions.length > 0 ? (
        <div>
          <h4 className="mb-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider">Contradictions</h4>
          {crossDoc.contradictions.map((c, i) => (
            <div key={i} className="mb-2 flex items-start gap-3 glass rounded-xl p-3.5">
              {c.severity === "CRITICAL" ? <XCircle className="mt-0.5 h-4 w-4 text-red-400 shrink-0" /> : <AlertTriangle className="mt-0.5 h-4 w-4 text-amber-400 shrink-0" />}
              <div>
                <p className="text-sm font-medium text-zinc-200">{c.field} <span className={cn("text-xs", c.severity === "CRITICAL" ? "text-red-400" : "text-amber-400")}>({c.severity})</span></p>
                <p className="font-mono text-xs text-zinc-500 mt-0.5">{c.values[0] ?? "?"} vs {c.values[1] ?? "?"}</p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 p-3.5">
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          <span className="text-sm font-medium text-emerald-300">No contradictions detected</span>
        </div>
      )}
      {crossDoc.entity_resolution && (
        <div>
          <h4 className="mb-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider">Entity Resolution</h4>
          <div className="grid grid-cols-3 gap-3">
            <MetricCard label="Match Score" value={crossDoc.entity_resolution.match_score.toFixed(2)} />
            <MetricCard label="Canonical Name" value={crossDoc.entity_resolution.canonical_name || "---"} />
            <MetricCard label="Same Person" value={crossDoc.entity_resolution.is_same_person ? "Yes" : "No"} />
          </div>
        </div>
      )}
    </div>
  );
}

function PolicyPanel({ policy }: { policy?: NonNullable<Job["result"]>["policy"] }) {
  if (!policy) return <Empty msg="No policy data available." />;
  return (
    <div className="space-y-5">
      <div className={cn(
        "inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold",
        policy.overall_status === "COMPLIANT" ? "bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20" : policy.overall_status === "NON_COMPLIANT" ? "bg-red-500/10 text-red-400 ring-1 ring-red-500/20" : "bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/20"
      )}>
        {policy.overall_status === "COMPLIANT" ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
        {policy.overall_status.replace(/_/g, " ")}
      </div>
      <div className="space-y-2">
        {policy.checks.map((c, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="flex items-start gap-3 glass rounded-xl p-3.5"
          >
            {c.status === "PASS" ? <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-400 shrink-0" /> : c.status === "FAIL" ? <XCircle className="mt-0.5 h-4 w-4 text-red-400 shrink-0" /> : <div className="mt-0.5 h-4 w-4 rounded-full border-2 border-zinc-600 shrink-0" />}
            <div>
              <p className="text-sm font-medium text-zinc-200">{c.requirement}</p>
              {c.explanation && <p className="text-xs text-zinc-500 mt-0.5">{c.explanation}</p>}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

function GaugeCard({ label, value, format, color }: { label: string; value: number; format: (v: number) => string; color: "emerald" | "blue" | "amber" | "red" }) {
  const pct = Math.min(100, value * 100);
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;
  const stroke = { emerald: "#10b981", blue: "#3b82f6", amber: "#f59e0b", red: "#ef4444" }[color];
  return (
    <div className="glass rounded-xl p-4 flex flex-col items-center">
      <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">{label}</p>
      <div className="relative h-20 w-20">
        <svg className="h-20 w-20 -rotate-90" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r={radius} fill="none" stroke="#27272a" strokeWidth="6" />
          <motion.circle
            cx="40" cy="40" r={radius} fill="none"
            stroke={stroke} strokeWidth="6" strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold text-white">{format(value)}</span>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass rounded-xl p-4 text-center">
      <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">{label}</p>
      <p className="mt-1.5 text-lg font-bold text-zinc-200">{value}</p>
    </div>
  );
}

function ProgressBar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="mb-1.5 flex justify-between text-xs">
        <span className="capitalize text-zinc-400 font-medium">{label}</span>
        <span className="font-mono text-zinc-500">{value.toFixed(2)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-zinc-800/80">
        <motion.div
          className={cn("h-full rounded-full", value > 0.7 ? "bg-red-500" : value > 0.4 ? "bg-amber-500" : "bg-emerald-500")}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, value * 100)}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-zinc-600">
      <FileSearch className="mb-3 h-10 w-10" />
      <p className="text-sm">{msg}</p>
    </div>
  );
}
