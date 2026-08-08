import { cn, statusColor, pipelineStatusLabel } from "@/lib/utils";

export function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium",
        statusColor(status)
      )}
    >
      {status}
    </span>
  );
}

export function PipelineStatusBadge({ status }: { status: string }) {
  const { label, color } = pipelineStatusLabel(status);
  return <span className={cn("text-xs font-semibold", color)}>{label}</span>;
}
