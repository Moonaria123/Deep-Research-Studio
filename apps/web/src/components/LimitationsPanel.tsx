import { useState } from "react";
import { ChevronDown, Info } from "lucide-react";
import type { Citation } from "@/lib/types";
import { AnswerBody } from "./AnswerBody";
import { cn } from "@/lib/utils";

/**
 * A low-emphasis, collapsible "Limitations and open questions" panel rendered below
 * the confident report body. Caveats live here — separated from the findings — so the
 * main answer reads as a briefing rather than a stream of disclaimers. Renders nothing
 * when there are no limitations, keeping older runs unchanged.
 */
export function LimitationsPanel({
  text,
  citations,
  defaultOpen = false,
}: {
  text?: string;
  citations: Citation[];
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  if (!text?.trim()) return null;

  return (
    <section
      aria-label="Limitations and open questions"
      className="rounded-lg border border-border bg-bg-subtle/60"
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left focus-ring"
      >
        <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
          <Info className="h-3.5 w-3.5" aria-hidden="true" />
          Limitations &amp; open questions
        </span>
        <ChevronDown
          className={cn("h-4 w-4 text-text-faint transition-transform", open && "rotate-180")}
          aria-hidden="true"
        />
      </button>
      {open && (
        <div className="border-t border-border px-4 py-3 text-text-muted">
          <AnswerBody text={text} citations={citations} />
        </div>
      )}
    </section>
  );
}
