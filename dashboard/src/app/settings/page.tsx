"use client";

import { useState, useEffect } from "react";
import {
  Save,
  Trash2,
  Server,
  Activity,
  Settings2,
  Database,
  CheckCircle2,
  WifiOff,
  Loader2,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { getHealth } from "@/lib/api";

interface Settings {
  apiUrl: string;
  autoRefreshInterval: number;
  defaultDocType: string;
}

const DEFAULT_SETTINGS: Settings = {
  apiUrl: "http://localhost:8000",
  autoRefreshInterval: 3000,
  defaultDocType: "auto",
};

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.06, duration: 0.4 },
  }),
};

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);
  const [apiStatus, setApiStatus] = useState<"connected" | "disconnected" | "checking">("checking");
  const [jobCount, setJobCount] = useState(0);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("kyc_settings");
      if (stored) setSettings(JSON.parse(stored));
    } catch {}

    try {
      const jobs = JSON.parse(localStorage.getItem("kyc_jobs") || "[]");
      setJobCount(jobs.length);
    } catch {}

    getHealth()
      .then(() => setApiStatus("connected"))
      .catch(() => setApiStatus("disconnected"));
  }, []);

  const handleSave = () => {
    localStorage.setItem("kyc_settings", JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleClearData = () => {
    if (confirm("Clear all stored job data? This cannot be undone.")) {
      localStorage.removeItem("kyc_jobs");
      setJobCount(0);
    }
  };

  const handleTestConnection = async () => {
    setApiStatus("checking");
    try {
      await getHealth();
      setApiStatus("connected");
    } catch {
      setApiStatus("disconnected");
    }
  };

  const StatusIcon =
    apiStatus === "connected" ? CheckCircle2 :
    apiStatus === "disconnected" ? WifiOff : Loader2;

  const statusColor =
    apiStatus === "connected" ? "text-emerald-400" :
    apiStatus === "disconnected" ? "text-red-400" : "text-zinc-500";

  const statusLabel =
    apiStatus === "connected" ? "Connected" :
    apiStatus === "disconnected" ? "Unreachable" : "Testing...";

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={0}>
        <h1 className="text-3xl font-bold tracking-tight text-white">Settings</h1>
        <p className="mt-1.5 text-sm text-zinc-500">Configure API connection and dashboard preferences</p>
      </motion.div>

      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={1} className="glass rounded-xl p-6">
        <div className="mb-5 flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10 ring-1 ring-blue-500/20">
            <Server className="h-4 w-4 text-blue-400" />
          </div>
          <h2 className="text-sm font-semibold text-zinc-200">API Connection</h2>
        </div>

        <div className="space-y-5">
          <div>
            <label className="mb-2 block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Backend URL</label>
            <div className="w-full rounded-xl bg-zinc-900/60 ring-1 ring-zinc-800/50 px-4 py-2.5 text-sm font-mono text-zinc-400">
              {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
            </div>
            <p className="mt-1.5 text-[11px] text-zinc-600">Set via NEXT_PUBLIC_API_URL environment variable</p>
          </div>

          <div className="flex items-center justify-between rounded-xl bg-zinc-900/40 ring-1 ring-zinc-800/30 px-4 py-3">
            <div className="flex items-center gap-3">
              <StatusIcon className={cn("h-4 w-4", statusColor, apiStatus === "checking" && "animate-spin")} />
              <div>
                <span className="text-sm text-zinc-400">Status: </span>
                <span className={cn("text-sm font-semibold", statusColor)}>{statusLabel}</span>
              </div>
            </div>
            <button
              onClick={handleTestConnection}
              className="rounded-lg glass glass-hover px-3.5 py-1.5 text-xs font-medium text-zinc-300 transition-all"
            >
              Test
            </button>
          </div>
        </div>
      </motion.div>

      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={2} className="glass rounded-xl p-6">
        <div className="mb-5 flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/10 ring-1 ring-amber-500/20">
            <Settings2 className="h-4 w-4 text-amber-400" />
          </div>
          <h2 className="text-sm font-semibold text-zinc-200">Preferences</h2>
        </div>

        <div className="space-y-5">
          <div>
            <label className="mb-2 block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
              Auto-Refresh Interval (ms)
            </label>
            <input
              type="number"
              value={settings.autoRefreshInterval}
              onChange={(e) =>
                setSettings({ ...settings, autoRefreshInterval: parseInt(e.target.value) || 3000 })
              }
              min={1000}
              max={30000}
              step={1000}
              className="w-full rounded-xl bg-zinc-900/60 ring-1 ring-zinc-800/50 px-4 py-2.5 text-sm text-zinc-200 outline-none transition-all focus:ring-blue-500/30"
            />
          </div>
          <div>
            <label className="mb-2 block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
              Default Document Type
            </label>
            <select
              value={settings.defaultDocType}
              onChange={(e) => setSettings({ ...settings, defaultDocType: e.target.value })}
              className="w-full rounded-xl bg-zinc-900/60 ring-1 ring-zinc-800/50 px-4 py-2.5 text-sm text-zinc-200 outline-none transition-all focus:ring-blue-500/30"
            >
              <option value="auto">Auto-detect</option>
              <option value="pan">PAN Card</option>
              <option value="aadhaar">Aadhaar Card</option>
              <option value="passport">Passport</option>
              <option value="driving_license">Driving License</option>
            </select>
          </div>
        </div>
      </motion.div>

      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={3} className="glass rounded-xl p-6">
        <div className="mb-5 flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-500/10 ring-1 ring-red-500/20">
            <Database className="h-4 w-4 text-red-400" />
          </div>
          <h2 className="text-sm font-semibold text-zinc-200">Data Management</h2>
        </div>

        <div className="flex items-center justify-between rounded-xl bg-zinc-900/40 ring-1 ring-zinc-800/30 px-4 py-3.5">
          <div>
            <p className="text-sm font-medium text-zinc-200">{jobCount} stored jobs</p>
            <p className="text-[11px] text-zinc-600">Stored in browser localStorage</p>
          </div>
          <button
            onClick={handleClearData}
            disabled={jobCount === 0}
            className={cn(
              "flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-all",
              "bg-red-500/10 ring-1 ring-red-500/20 text-red-400 hover:bg-red-500/20",
              "disabled:opacity-40 disabled:pointer-events-none"
            )}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear All
          </button>
        </div>
      </motion.div>

      <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={4} className="flex justify-end">
        <button
          onClick={handleSave}
          className={cn(
            "flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-medium transition-all duration-200",
            saved
              ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20"
              : "bg-blue-600 text-white hover:bg-blue-500 active:scale-[0.98] shadow-lg shadow-blue-600/20"
          )}
        >
          {saved ? <CheckCircle2 className="h-4 w-4" /> : <Save className="h-4 w-4" />}
          {saved ? "Saved!" : "Save Settings"}
        </button>
      </motion.div>
    </div>
  );
}
