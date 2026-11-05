import { useMemo } from "react";
import { FileText, Gauge, Search, Repeat } from "lucide-react";
import type { BudgetSnapshot, Mode, RunEvent } from "@/lib/types";

/** Pull the most recent ``budget_snapshot`` payload out of the event stream. */
export function latestBudgetSnapshot(events: RunEvent[]): BudgetSnapshot | undefined {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (events[i].type === "budget_snapshot") {
      return events[i].payload as unknown as BudgetSnapshot;
    }
  }
  return undefined;
}

const COUNCIL_STAGE_LABELS: Record<string, string> = {
  search: "Searching shared evidence",
  drafts: "Drafting member answers",
  critique: "Peer critique round",
  vote: "Selecting best draft",
  synthesis: "Chair synthesis",
};

const RESEARCH_STAGE_LABELS: Record<string, string> = {
  source_recovery: "Verifying better sources",
  answer_repair: "Repairing source citations",
};

function displayStage(mode: Mode, stage: string): string {
  if (mode === "council") return COUNCIL_STAGE_LABELS[stage] ?? stage;
  return RESEARCH_STAGE_LABELS[stage] ?? stage;
}

function tierLabel(tier?: string): string {
  return tier ? tier.replace(/_/g, " ") : "";
}

interface RunBudgetStripProps {
  events: RunEvent[];
  mode: Mode;
  /** The run's resolved budget (from the run record), used before any snapshot arrives. */
  budget?: Record<string, number>;
  /** Final gap-loop count from the completed result (deep mode). */
  gapRounds?: number;
  recoverySearchesMax?: number;
  recoverySearchesUsed?: number;
}

/**
 * A compact strip showing the run's search budget consumption: searches used vs.
 * max, the current pipeline stage while streaming, and (for deep runs) the number
 * of gap-loop rounds. Renders nothing when there is no budget context to show.
 */
export function RunBudgetStrip({
  events,
  mode,
  budget,
  gapRounds,
  recoverySearchesMax,
  recoverySearchesUsed,
}: RunBudgetStripProps) {
  const snapshot = useMemo(() => latestBudgetSnapshot(events), [events]);

  const max = snapshot?.searches_max ?? budget?.max_searches;
  if (max == null) return null;
  const used = snapshot?.searches_used ?? 0;
  // The completed result's gap_rounds is authoritative; fall back to the live snapshot.
  const gaps = gapRounds ?? snapshot?.gap_rounds;
  const stage = snapshot?.stage;
  const stageLabel = stage ? displayStage(mode, stage) : "";
  const remaining = snapshot?.remaining_searches;
  const pagesFetched = snapshot?.pages_fetched;
  const tier = tierLabel(snapshot?.search_tier);
  const recoveryMax = recoverySearchesMax ?? snapshot?.recovery_searches_max;
  const recoveryUsed = recoverySearchesUsed ?? snapshot?.recovery_searches_used ?? 0;

  return (
    <div
      aria-label="Run budget"
      className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border bg-bg-subtle/60 px-3 py-2 text-xs text-text-muted"
    >
      <span className="flex items-center gap-1.5 font-medium text-text-body">
        <Gauge className="h-3.5 w-3.5" aria-hidden="true" />
        Budget
      </span>
      <span className="flex items-center gap-1.5">
        <Search className="h-3.5 w-3.5" aria-hidden="true" />
        Searches <span className="font-medium text-text-body">{used}/{max}</span>
      </span>
      {recoveryMax != null && recoveryMax > 0 && (
        <span className="flex items-center gap-1.5">
          <Repeat className="h-3.5 w-3.5" aria-hidden="true" />
          Recovery searches <span className="font-medium text-text-body">{recoveryUsed}/{recoveryMax}</span>
        </span>
      )}
      {remaining != null && <span className="font-medium text-text-body">{remaining} left</span>}
      {pagesFetched != null && (
        <span className="flex items-center gap-1.5">
          <FileText className="h-3.5 w-3.5" aria-hidden="true" />
          <span className="font-medium text-text-body">{pagesFetched} pages</span>
        </span>
      )}
      {tier && (
        <span className="rounded bg-surface px-1.5 py-0.5 text-[11px] text-text-body">
          <span>{tier}</span>
          {snapshot?.degraded_fetch ? <span> · degraded fetch</span> : null}
        </span>
      )}
      {mode === "deep" && gaps != null && (
        <span className="flex items-center gap-1.5">
          <Repeat className="h-3.5 w-3.5" aria-hidden="true" />
          Gap rounds <span className="font-medium text-text-body">{gaps}</span>
        </span>
      )}
      {stageLabel && (
        <span className="flex items-center gap-1.5">
          Stage <span className="rounded bg-surface px-1.5 py-0.5 text-[11px] text-text-body">{stageLabel}</span>
        </span>
      )}
    </div>
  );
}
