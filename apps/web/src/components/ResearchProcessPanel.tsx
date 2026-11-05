import { useState } from "react";
import { ChevronDown, FlaskConical } from "lucide-react";
import type { GroundedAnswer } from "@/lib/types";
import { cn } from "@/lib/utils";
import { SectionCard } from "./SectionCard";

/**
 * A default-collapsed "Research process" appendix for Deep Research runs. The final
 * synthesized report is the primary deliverable; the intermediate per-section drafts
 * live here so the main view reads as one cohesive report instead of duplicating the
 * pipeline output. Renders nothing when there are no section drafts.
 */
export function ResearchProcessPanel({
  sections,
  defaultOpen = false,
}: {
  sections: { title: string; answer: GroundedAnswer }[];
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  if (!sections.length) return null;

  return (
    <section aria-label="Research process" className="rounded-lg border border-border bg-bg-subtle/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left focus-ring"
      >
        <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
          <FlaskConical className="h-3.5 w-3.5" aria-hidden="true" />
          Research process
          <span className="text-text-faint">({sections.length})</span>
        </span>
        <ChevronDown
          className={cn("h-4 w-4 text-text-faint transition-transform", open && "rotate-180")}
          aria-hidden="true"
        />
      </button>
      {open && (
        <div className="space-y-3 border-t border-border px-4 py-3">
          {sections.map((s, i) => (
            <SectionCard key={s.title} index={i + 1} title={s.title} answer={s.answer} defaultOpen={false} />
          ))}
        </div>
      )}
    </section>
  );
}
