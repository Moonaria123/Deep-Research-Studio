import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import type { RunEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Render node_status events as a vertical stage timeline with an active pulse. */
export function ProgressTimeline({ events, done }: { events: RunEvent[]; done: boolean }) {
  const stages = events
    .filter((e) => e.type === "node_status")
    .map((e) => {
      const payload = e.payload as { node?: string; message?: string };
      const node = String(payload.node ?? "step");
      return { node, label: payload.message?.trim() || stageLabel(node) };
    });
  const unique = [...new Map(stages.map((stage) => [stage.node, stage])).values()];
  const activeIndex = done ? unique.length : unique.length - 1;

  return (
    <ol className="space-y-3" aria-label="Run progress">
      {unique.map(({ node, label }, i) => {
        const state = i < activeIndex ? "done" : i === activeIndex && !done ? "active" : "done";
        return (
          <li key={node} data-anim className="flex items-center gap-3">
            {state === "active" ? (
              <Loader2 className="h-4 w-4 animate-spin text-accent" aria-hidden />
            ) : state === "done" ? (
              <CheckCircle2 className="h-4 w-4 text-success" aria-hidden />
            ) : (
              <Circle className="h-4 w-4 text-text-faint" aria-hidden />
            )}
            <span
              className={cn(
                "text-sm capitalize",
                state === "active" ? "font-medium text-text" : "text-text-muted",
              )}
            >
              {label}
            </span>
          </li>
        );
      })}
      {unique.length === 0 && <li className="text-sm text-text-muted">Starting…</li>}
    </ol>
  );
}

function stageLabel(node: string): string {
  if (node === "source_recovery") return "Verifying better sources…";
  if (node === "answer_repair") return "Repairing source citations…";
  return node.replace(/_/g, " ");
}
