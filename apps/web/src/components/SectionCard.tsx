import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { GroundedAnswer } from "@/lib/types";
import { cn } from "@/lib/utils";
import { AnswerBody } from "./AnswerBody";

export function SectionCard({
  index,
  title,
  answer,
  defaultOpen = true,
}: {
  index: number;
  title: string;
  answer: GroundedAnswer;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const headingId = `section-${index}`;
  return (
    <section data-anim className="rounded-md border border-border bg-surface shadow-card">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={`${headingId}-body`}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 px-5 py-4 text-left focus-ring"
      >
        <span className="text-xs font-semibold text-text-faint">{String(index).padStart(2, "0")}</span>
        <h3 id={headingId} className="flex-1 text-base font-semibold text-text">
          {title}
        </h3>
        <ChevronDown className={cn("h-4 w-4 text-text-muted transition-transform", open && "rotate-180")} aria-hidden />
      </button>
      {open && (
        <div id={`${headingId}-body`} className="px-5 pb-5">
          <AnswerBody text={answer.text} citations={answer.citations} />
        </div>
      )}
    </section>
  );
}
