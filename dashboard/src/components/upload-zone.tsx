"use client";

import { useCallback, useState } from "react";
import { Upload, FileImage, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface UploadZoneProps {
  onUpload: (file: File) => Promise<string>;
  disabled?: boolean;
}

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"];

export function UploadZone({ onUpload, disabled }: UploadZoneProps) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0 || disabled) return;
      setError(null);
      setSuccess(null);

      const valid: File[] = [];
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        if (!ALLOWED_TYPES.includes(f.type) && !f.name.match(/\.(jpe?g|png|webp|tiff?|bmp)$/i)) {
          setError(`"${f.name}" is not a supported image format.`);
          return;
        }
        if (f.size > MAX_FILE_SIZE) {
          setError(`"${f.name}" exceeds the 20 MB size limit.`);
          return;
        }
        valid.push(f);
      }

      setUploading(true);
      try {
        for (const f of valid) {
          await onUpload(f);
        }
        setSuccess(`${valid.length} document${valid.length > 1 ? "s" : ""} submitted for processing`);
        setTimeout(() => setSuccess(null), 4000);
      } finally {
        setUploading(false);
      }
    },
    [onUpload, disabled]
  );

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
        className={cn(
          "relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-16 transition-all duration-300",
          dragging
            ? "border-blue-400 bg-blue-500/5 shadow-[0_0_40px_rgba(59,130,246,0.1)]"
            : "border-zinc-700/50 hover:border-zinc-600",
          disabled && "pointer-events-none opacity-50"
        )}
      >
        <div className={cn(
          "flex h-16 w-16 items-center justify-center rounded-2xl transition-all duration-300",
          dragging ? "bg-blue-500/10 scale-110" : "bg-zinc-800/50"
        )}>
          {uploading ? (
            <Loader2 className="h-8 w-8 animate-spin text-blue-400" />
          ) : (
            <Upload className={cn("h-8 w-8 transition-colors", dragging ? "text-blue-400" : "text-zinc-600")} />
          )}
        </div>
        <p className="mt-5 text-sm font-medium text-zinc-300">
          {uploading ? "Submitting to pipeline..." : "Drop document images here or click to upload"}
        </p>
        <p className="mt-1.5 text-xs text-zinc-600">
          PAN, Aadhaar, Passport, DL, Voter ID &middot; JPEG, PNG, WebP, TIFF &middot; Max 20 MB
        </p>
        <label className={cn(
          "mt-5 inline-flex cursor-pointer items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-medium transition-all duration-200",
          "bg-blue-600 text-white hover:bg-blue-500 active:scale-[0.98] shadow-lg shadow-blue-600/20",
          (disabled || uploading) && "pointer-events-none opacity-50"
        )}>
          <FileImage className="h-4 w-4" />
          Choose Files
          <input
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
            disabled={disabled || uploading}
          />
        </label>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="flex items-center gap-2 rounded-xl bg-red-500/10 border border-red-500/20 px-4 py-3"
          >
            <AlertCircle className="h-4 w-4 text-red-400 shrink-0" />
            <p className="text-sm text-red-300">{error}</p>
          </motion.div>
        )}
        {success && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="flex items-center gap-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 px-4 py-3"
          >
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
            <p className="text-sm text-emerald-300">{success}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
