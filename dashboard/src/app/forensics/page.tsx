"use client";

import { useState, useEffect, useMemo } from "react";
import {
  ShieldAlert,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Fingerprint,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn, riskColor, formatConfidence } from "@/lib/utils";
import type { Job, ForensicsResult } from "@/types/api";

interface ForensicsCase {
  job_id: string;
  filename: string;
  forensics: ForensicsResult;
}

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.06, duration: 0.4 },
  }),
};

const RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"] as const;

const riskMeta: Record<string, { icon: typeof CheckCircle; bg: string; ring: string; glow: string }> = {
  LOW: { icon: CheckCircle, bg: "bg-emerald-500/10", ring: "ring-emerald-500/20", glow: "shadow-emerald-500/5" },
  MEDIUM: { icon: AlertTriangle, bg: "bg-amber-500/10", ring: "ring-amber-500/20", glow: "shadow-amber-500/5" },
  HIGH: { icon: XCircle, bg: "bg-red-500/10", ring: "ring-red-500/20", glow: "shadow-red-500/5" },
  CRITICAL: { icon: ShieldAlert, bg: "bg-red-600/10", ring: "ring-red-600/20", glow: "shadow-red-600/5" },
};

function SpoofGauge({ score, size = 56 }: { score: number; size?: number }) {
  const r = (size - 6) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - score);
  const color = score > 0.7 ? "#ef4444" : score > 0.4 ? "#f59e0b" : "#10b981";

  return (
    <svg width={size} height={size} className="flex-shrink-0 -rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#27272a" strokeWidth={5} />
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={5}
        strokeLinecap="round"
        strokeDasharray={circumference}
        initial={{ strokeDashoffset: circumference }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 1, ease: "easeOut" }}
      />
      <text
        x={size / 2}
        y={size / 2}
        textAnchor="middle"
        dominantBaseline="central"
        className="rotate-90 origin-center fill-zinc-200 text-[11px] font-mono font-bold"
      >
        {(score * 100).toFixed(0)}
      </text>
    </svg>
  );
}

export default function ForensicsPage() {
  const [cases, setCases] = useState<ForensicsCase[]>([]);
  const [selectedRisk, setSelectedRisk] = useState<string>("ALL");

  const load = () => {
    try {
      const stored = JSON.parse(localStorage.getItem("kyc_jobs") || "[]") as Job[];
      const withForensics = stored
        .filter(
          (j): j is Job & { result: { forensics: ForensicsResult } } =>
            !!j.result?.forensics
        )
        .map((j) => ({
          job_id: j.job_id,
          filename: j.filename,
          forensics: j.result.forensics,
        }));
      setCases(withForensics);
    } catch {
      setCases([]);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const filtered = useMemo(() => {
    if (selectedRisk === "ALL") return cases;
    return cases.filter((c) => c.forensics.risk_level === selectedRisk);
  }, [cases, selectedRisk]);

  const riskCounts = useMemo(() => {
    const counts: Record<string, number> = { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0 };
    cases.forEach((c) => {
      if (c.forensics.risk_level in counts)
        counts[c.forensics.risk_level]++;
    });
    return counts;
  }, [cases]);

  return (
    <div className="space-y-6">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        custom={0}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Forensics</h1>
          <p className="mt-1.5 text-sm text-zinc-500">
            Document tampering detection — ELA, copy-move, font analysis, Moire detection
          </p>
        </div>
        <button
          onClick={load}
          className="flex items-center gap-2 rounded-xl glass glass-hover px-4 py-2.5 text-sm font-medium text-zinc-300"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </motion.div>

      <div className="grid gap-4 sm:grid-cols-4">
        {RISK_LEVELS.map((level, i) => {
          const meta = riskMeta[level];
          const Icon = meta.icon;
          const active = selectedRisk === level;
          return (
            <motion.button
              key={level}
              initial="hidden"
              animate="visible"
              variants={fadeUp}
              custom={i + 1}
              onClick={() => setSelectedRisk(active ? "ALL" : level)}
              className={cn(
                "glass rounded-xl p-4 text-left transition-all duration-200",
                active
                  ? "ring-1 ring-blue-500/30 shadow-lg shadow-blue-500/5"
                  : "glass-hover"
              )}
            >
              <div className="flex items-center justify-between">
                <span className={cn("text-xs font-semibold uppercase tracking-wider", riskColor(level))}>{level}</span>
                <div className={cn("flex h-7 w-7 items-center justify-center rounded-lg ring-1", meta.bg, meta.ring)}>
                  <Icon className={cn("h-3.5 w-3.5", riskColor(level))} />
                </div>
              </div>
              <p className="mt-2.5 text-3xl font-bold tracking-tight text-white">{riskCounts[level]}</p>
              <p className="text-[11px] text-zinc-600">documents</p>
            </motion.button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        {filtered.length === 0 ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass rounded-2xl p-16 text-center"
          >
            <Fingerprint className="mx-auto h-10 w-10 text-zinc-800" />
            <p className="mt-4 text-zinc-600">
              {cases.length === 0
                ? "No forensics data yet. Process documents to see analysis."
                : "No documents match the selected risk level."}
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="list"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {filtered.map((c, idx) => {
              const meta = riskMeta[c.forensics.risk_level] || riskMeta.LOW;
              const Icon = meta.icon;
              return (
                <motion.div
                  key={c.job_id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="glass rounded-xl p-5"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-4">
                      <SpoofGauge score={c.forensics.spoof_score} />
                      <div>
                        <p className="text-sm font-medium text-zinc-200">{c.filename}</p>
                        <p className="mt-0.5 font-mono text-xs text-zinc-600">{c.job_id.slice(0, 12)}...</p>
                      </div>
                    </div>
                    <div className={cn(
                      "flex items-center gap-2 rounded-lg px-3 py-1.5 ring-1",
                      meta.bg, meta.ring
                    )}>
                      <Icon className={cn("h-3.5 w-3.5", riskColor(c.forensics.risk_level))} />
                      <span className={cn("text-xs font-semibold", riskColor(c.forensics.risk_level))}>
                        {c.forensics.risk_level}
                      </span>
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="mb-1.5 flex items-center justify-between text-xs">
                      <span className="text-zinc-500">Spoof Score</span>
                      <span className={riskColor(c.forensics.risk_level)}>
                        {formatConfidence(c.forensics.spoof_score)}
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-zinc-800/80">
                      <motion.div
                        className="h-full rounded-full"
                        style={{
                          background:
                            c.forensics.spoof_score > 0.7
                              ? "#ef4444"
                              : c.forensics.spoof_score > 0.4
                                ? "#f59e0b"
                                : "#10b981",
                        }}
                        initial={{ width: 0 }}
                        animate={{ width: `${c.forensics.spoof_score * 100}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                      />
                    </div>
                  </div>

                  {Object.keys(c.forensics.component_scores).length > 0 && (
                    <div className="mt-4 grid gap-2 sm:grid-cols-3 lg:grid-cols-5">
                      {Object.entries(c.forensics.component_scores).map(([key, score], si) => (
                        <motion.div
                          key={key}
                          initial={{ opacity: 0, y: 6 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.3 + si * 0.04 }}
                          className="rounded-lg bg-zinc-900/60 ring-1 ring-zinc-800/50 px-3 py-2"
                        >
                          <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">
                            {key.replace(/_/g, " ")}
                          </p>
                          <p className={cn(
                            "mt-0.5 font-mono text-sm font-bold",
                            score > 0.5 ? "text-red-400" : score > 0.3 ? "text-amber-400" : "text-emerald-400"
                          )}>
                            {formatConfidence(score)}
                          </p>
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {c.forensics.evidence.length > 0 && (
                    <div className="mt-4 space-y-1.5">
                      <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Evidence</p>
                      {c.forensics.evidence.map((ev, ei) => (
                        <motion.div
                          key={ei}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.4 + ei * 0.04 }}
                          className="flex items-center gap-3 rounded-lg bg-zinc-900/40 ring-1 ring-zinc-800/30 px-3 py-2 text-xs"
                        >
                          <span className="rounded-md bg-zinc-800/80 px-2 py-0.5 font-mono text-[10px] text-zinc-400 uppercase">
                            {ev.type}
                          </span>
                          <span className="flex-1 text-zinc-400">{ev.detail}</span>
                          <span className={cn(
                            "font-mono font-bold",
                            ev.score > 0.5 ? "text-red-400" : ev.score > 0.3 ? "text-amber-400" : "text-zinc-600"
                          )}>
                            {formatConfidence(ev.score)}
                          </span>
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {c.forensics.recommendation && (
                    <p className="mt-3 text-xs italic text-zinc-500">
                      {c.forensics.recommendation}
                    </p>
                  )}
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
