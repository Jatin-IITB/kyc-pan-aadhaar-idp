"use client";

import { useEffect, useState } from "react";
import {
  FileCheck,
  ShieldAlert,
  Clock,
  TrendingUp,
  Upload,
  Activity,
  ArrowRight,
  Zap,
  Brain,
  Eye,
  Scale,
  Search,
} from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";
import { getHealth } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/api";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
  }),
};

function useJobStats() {
  const [stats, setStats] = useState({ total: 0, succeeded: 0, failed: 0, autoClear: 0, avgConf: 0, avgSpoof: 0 });
  useEffect(() => {
    try {
      const jobs: Job[] = JSON.parse(localStorage.getItem("kyc_jobs") || "[]");
      const completed = jobs.filter((j) => j.status === "SUCCEEDED" && j.result);
      const autoClear = completed.filter((j) => j.result?.status === "AUTO_CLEARED").length;
      const confs = completed.map((j) => j.result?.calibration?.calibrated_confidence ?? 0).filter(Boolean);
      const spoofs = completed.map((j) => j.result?.forensics?.spoof_score ?? 0).filter(Boolean);
      setStats({
        total: jobs.length,
        succeeded: completed.length,
        failed: jobs.filter((j) => j.status === "FAILED").length,
        autoClear: completed.length > 0 ? (autoClear / completed.length) * 100 : 0,
        avgConf: confs.length > 0 ? confs.reduce((a, b) => a + b, 0) / confs.length * 100 : 0,
        avgSpoof: spoofs.length > 0 ? spoofs.reduce((a, b) => a + b, 0) / spoofs.length * 100 : 0,
      });
    } catch { /* empty */ }
  }, []);
  return stats;
}

export default function DashboardPage() {
  const [apiStatus, setApiStatus] = useState<"connected" | "disconnected" | "checking">("checking");
  const stats = useJobStats();

  useEffect(() => {
    getHealth()
      .then(() => setApiStatus("connected"))
      .catch(() => setApiStatus("disconnected"));
  }, []);

  const STATS = [
    { label: "Documents Processed", value: String(stats.total || "0"), icon: FileCheck, color: "text-blue-400", bg: "bg-blue-500/10", ring: "ring-blue-500/20" },
    { label: "Success Rate", value: stats.total > 0 ? `${((stats.succeeded / stats.total) * 100).toFixed(0)}%` : "---", icon: TrendingUp, color: "text-emerald-400", bg: "bg-emerald-500/10", ring: "ring-emerald-500/20" },
    { label: "Avg Confidence", value: stats.avgConf > 0 ? `${stats.avgConf.toFixed(1)}%` : "---", icon: ShieldAlert, color: "text-amber-400", bg: "bg-amber-500/10", ring: "ring-amber-500/20" },
    { label: "Avg Spoof Score", value: stats.avgSpoof > 0 ? `${stats.avgSpoof.toFixed(1)}%` : "---", icon: Clock, color: "text-red-400", bg: "bg-red-500/10", ring: "ring-red-500/20" },
  ];

  return (
    <div className="space-y-8">
      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={0}>
        <h1 className="text-3xl font-bold tracking-tight text-white">Dashboard</h1>
        <p className="mt-1.5 text-sm text-zinc-500">
          Multi-agent document intelligence platform
        </p>
      </motion.div>

      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={1}>
        <div className={cn(
          "flex items-center gap-3 rounded-xl px-4 py-3 text-sm transition-all duration-300",
          apiStatus === "connected"
            ? "glass glow-emerald text-emerald-400"
            : apiStatus === "disconnected"
              ? "glass glow-red text-red-400"
              : "glass text-zinc-400"
        )}>
          <div className="relative">
            <div className={cn(
              "h-2 w-2 rounded-full",
              apiStatus === "connected" ? "bg-emerald-400" : apiStatus === "disconnected" ? "bg-red-400" : "bg-zinc-500"
            )} />
            {apiStatus === "connected" && <div className="absolute inset-0 h-2 w-2 animate-ping rounded-full bg-emerald-400/50" />}
          </div>
          <Activity className="h-4 w-4" />
          <span className="font-medium">
            {apiStatus === "connected"
              ? "Pipeline Online"
              : apiStatus === "disconnected"
                ? "Pipeline Offline"
                : "Connecting..."}
          </span>
          {apiStatus === "connected" && (
            <span className="ml-auto text-xs text-emerald-500/70">localhost:8000</span>
          )}
        </div>
      </motion.div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STATS.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.label}
              initial="hidden"
              animate="visible"
              variants={fadeUp}
              custom={i + 2}
              className="glass glass-hover rounded-xl p-5"
            >
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium tracking-wide text-zinc-500 uppercase">{stat.label}</p>
                <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg ring-1", stat.bg, stat.ring)}>
                  <Icon className={cn("h-4 w-4", stat.color)} />
                </div>
              </div>
              <p className="mt-3 text-3xl font-bold tracking-tight text-white">{stat.value}</p>
            </motion.div>
          );
        })}
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {[
          { href: "/upload", label: "Upload Documents", desc: "PAN, Aadhaar, Passport, DL, and 8+ more", icon: Upload, accent: "blue" },
          { href: "/cases", label: "View Cases", desc: "Track verification status and decisions", icon: FileCheck, accent: "emerald" },
          { href: "/forensics", label: "Forensics Lab", desc: "ELA, copy-move, font & Moire analysis", icon: ShieldAlert, accent: "red" },
        ].map((item, i) => {
          const Icon = item.icon;
          const colors = {
            blue: "hover:border-blue-500/30 hover:shadow-[0_0_30px_rgba(59,130,246,0.08)]",
            emerald: "hover:border-emerald-500/30 hover:shadow-[0_0_30px_rgba(16,185,129,0.08)]",
            red: "hover:border-red-500/30 hover:shadow-[0_0_30px_rgba(239,68,68,0.08)]",
          };
          const iconColors = { blue: "text-blue-400 bg-blue-500/10", emerald: "text-emerald-400 bg-emerald-500/10", red: "text-red-400 bg-red-500/10" };
          return (
            <motion.div key={item.href} initial="hidden" animate="visible" variants={fadeUp} custom={i + 6}>
              <Link
                href={item.href}
                className={cn("group flex items-center gap-4 rounded-xl glass glass-hover p-5 transition-all duration-300", colors[item.accent as keyof typeof colors])}
              >
                <div className={cn("flex h-11 w-11 items-center justify-center rounded-xl", iconColors[item.accent as keyof typeof iconColors])}>
                  <Icon className="h-5 w-5" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-zinc-200 group-hover:text-white transition-colors">{item.label}</p>
                  <p className="text-xs text-zinc-600">{item.desc}</p>
                </div>
                <ArrowRight className="h-4 w-4 text-zinc-700 group-hover:text-zinc-400 group-hover:translate-x-0.5 transition-all" />
              </Link>
            </motion.div>
          );
        })}
      </div>

      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={9} className="glass rounded-xl p-6">
        <div className="flex items-center gap-2 mb-5">
          <Brain className="h-4 w-4 text-blue-400" />
          <h2 className="text-sm font-semibold tracking-wide text-zinc-300 uppercase">Pipeline Architecture</h2>
        </div>
        <div className="flex flex-wrap items-center gap-x-1 gap-y-2">
          {[
            { step: "Ingest", icon: Upload },
            { step: "Quality Gate", icon: Eye },
            { step: "Classify", icon: Search },
            { step: "YOLO Extract", icon: Zap },
            { step: "VLM Fallback", icon: Brain },
            { step: "Ensemble", icon: Scale },
            { step: "Validate", icon: FileCheck },
            { step: "LLM Rescue", icon: Brain },
            { step: "Forensics", icon: ShieldAlert },
            { step: "Policy (RAG)", icon: Activity },
            { step: "Cross-Doc", icon: Search },
            { step: "Calibrate", icon: Scale },
            { step: "Decide", icon: TrendingUp },
            { step: "Audit", icon: FileCheck },
          ].map(({ step, icon: StepIcon }, i, arr) => (
            <span key={step} className="flex items-center gap-1">
              <span className="inline-flex items-center gap-1.5 rounded-lg bg-zinc-800/60 px-2.5 py-1.5 text-xs font-mono text-zinc-400 ring-1 ring-zinc-700/50">
                <StepIcon className="h-3 w-3 text-zinc-600" />
                {step}
              </span>
              {i < arr.length - 1 && <span className="text-zinc-700 mx-0.5">&#8594;</span>}
            </span>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
