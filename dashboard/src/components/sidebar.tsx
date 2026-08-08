"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Upload,
  FileSearch,
  Shield,
  Activity,
  Settings,
  Menu,
  X,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getHealth } from "@/lib/api";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/cases", label: "Cases", icon: FileSearch },
  { href: "/forensics", label: "Forensics", icon: Shield },
  { href: "/metrics", label: "Metrics", icon: Activity },
  { href: "/settings", label: "Settings", icon: Settings },
] as const;

export function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const check = () => {
      getHealth()
        .then(() => { if (!cancelled) setApiUp(true); })
        .catch(() => { if (!cancelled) setApiUp(false); });
    };
    check();
    const id = setInterval(check, 15000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="fixed left-4 top-4 z-50 rounded-xl glass p-2.5 text-zinc-400 hover:text-white transition-colors md:hidden"
        aria-label="Open navigation"
      >
        <Menu className="h-5 w-5" />
      </button>

      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm md:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      <aside
        className={cn(
          "fixed left-0 top-0 z-50 flex h-screen w-64 flex-col border-r border-zinc-800/50 bg-zinc-950/95 backdrop-blur-xl transition-transform duration-300 ease-out md:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex h-16 items-center justify-between border-b border-zinc-800/50 px-5">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10 ring-1 ring-blue-500/20">
              <Zap className="h-4 w-4 text-blue-400" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-white">KYC Intelligence</h1>
              <p className="text-[10px] font-medium text-zinc-600">v1.0 &middot; Multi-Agent</p>
            </div>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="rounded-lg p-1 text-zinc-500 hover:text-white transition-colors md:hidden"
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <nav className="flex-1 space-y-0.5 px-3 py-4">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13px] font-medium transition-all duration-200",
                  active
                    ? "bg-blue-500/10 text-blue-400"
                    : "text-zinc-500 hover:bg-zinc-800/40 hover:text-zinc-200"
                )}
              >
                {active && (
                  <div className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-blue-400" />
                )}
                <Icon className={cn("h-4 w-4 transition-colors", active ? "text-blue-400" : "text-zinc-600 group-hover:text-zinc-400")} />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-zinc-800/50 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="relative">
              <div
                className={cn(
                  "h-2 w-2 rounded-full",
                  apiUp === null ? "bg-zinc-600" : apiUp ? "bg-emerald-400 pulse-dot pulse-dot-green" : "bg-red-400 pulse-dot pulse-dot-red"
                )}
              />
            </div>
            <span className="text-[11px] font-medium text-zinc-600">
              {apiUp === null ? "Connecting..." : apiUp ? "API Connected" : "API Offline"}
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}
