import { ShieldCheck, Scale, HelpCircle } from "lucide-react";
import { cn } from "@/lib/utils";

type Meta = { label: string; hint: string; cls: string; Icon: typeof ShieldCheck };

// Calm, non-alarming presentation: a confidence grade is informational, never a
// warning. Colors stay muted so an "uncertain" finding does not read as an error.
const META: Record<string, Meta> = {
  established: {
    label: "Established",
    hint: "Well supported by the cited sources",
    cls: "border-transparent bg-accent-soft text-success",
    Icon: ShieldCheck,
  },
  contested: {
    label: "Contested",
    hint: "Sources or models diverge — see limitations",
    cls: "border-transparent bg-[color:var(--warning)]/15 text-warning",
    Icon: Scale,
  },
  uncertain: {
    label: "Uncertain",
    hint: "Limited evidence — treat as tentative",
    cls: "border-border bg-surface-muted text-text-muted",
    Icon: HelpCircle,
  },
};

/**
 * A small qualitative confidence chip (Established / Contested / Uncertain) shown
 * instead of a bare coverage percentage. Renders nothing for unknown/empty labels
 * so older runs without a label degrade gracefully.
 */
export function ConfidenceTag({ label, className }: { label?: string; className?: string }) {
  const meta = label ? META[label] : undefined;
  if (!meta) return null;
  const { Icon } = meta;
  return (
    <span
      title={meta.hint}
      aria-label={`Confidence: ${meta.label}. ${meta.hint}`}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        meta.cls,
        className,
      )}
    >
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {meta.label}
    </span>
  );
}
