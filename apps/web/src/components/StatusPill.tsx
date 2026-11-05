import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

/** Top-bar status pill with an optional connected dot (paired with a text label for a11y). */
export function StatusPill({
  icon: Icon,
  label,
  compactLabel,
  connected,
  className,
  indicatorClassName,
}: {
  icon?: LucideIcon;
  label: string;
  compactLabel?: string;
  connected?: boolean;
  className?: string;
  indicatorClassName?: string;
}) {
  const displayLabel = compactLabel ?? label;
  return (
    <span
      className={cn(
        "inline-flex min-w-0 items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-text-muted",
        className,
      )}
      title={displayLabel !== label ? label : undefined}
      aria-label={label}
    >
      {Icon && <Icon className="h-3.5 w-3.5" aria-hidden />}
      <span className="truncate">{displayLabel}</span>
      {connected !== undefined && (
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            connected ? "bg-success" : "bg-text-faint",
            indicatorClassName,
          )}
          aria-hidden
        />
      )}
    </span>
  );
}
