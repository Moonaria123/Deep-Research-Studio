import { useState } from "react";
import { ChevronDown, Search } from "lucide-react";
import type { EvidenceSource } from "@/lib/types";
import { cn } from "@/lib/utils";
import { SourceCard } from "./SourceCard";
import { evidenceKey } from "@/lib/evidenceRoutes";

interface Lane {
  query: string;
  sources: EvidenceSource[];
}

function LaneRow({ lane, index }: { lane: Lane; index: number }) {
  const [open, setOpen] = useState(false);
  const count = lane.sources.length;
  return (
    <div className="rounded-md border border-border bg-surface">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left focus-ring"
      >
        <span className="flex min-w-0 items-center gap-2">
          <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-accent-soft text-[11px] font-semibold text-accent">
            {index + 1}
          </span>
          <span className="truncate text-sm text-text-body">{lane.query}</span>
        </span>
        <span className="flex shrink-0 items-center gap-2">
          <span className="rounded-full bg-bg-subtle px-2 py-0.5 text-xs text-text-muted">
            {count} source{count === 1 ? "" : "s"}
          </span>
          <ChevronDown
            className={cn("h-4 w-4 text-text-faint transition-transform", open && "rotate-180")}
            aria-hidden="true"
          />
        </span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-border px-3 py-3">
          {count === 0 ? (
            <p className="text-xs text-text-faint">No sources retrieved for this lane.</p>
          ) : (
            lane.sources.map((source, i) => (
              <SourceCard key={`${source.document_id}-${i}`} index={i + 1} source={source} />
            ))
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Makes the Multiple-mode search strategy transparent: each parallel lane's
 * sub-query is listed with its retrieved-source count and can be expanded to
 * inspect that lane's sources. The merged `SourcesRail` stays unchanged.
 * Renders nothing when there are no lanes.
 */
export function MultipleLanesPanel({
  lanes,
  adoptedSources,
}: {
  lanes: Lane[];
  adoptedSources?: EvidenceSource[];
}) {
  const adopted = adoptedSources ? new Set(adoptedSources.map(evidenceKey)) : null;
  const visibleLanes = adopted
    ? lanes.map((lane) => ({
        ...lane,
        sources: lane.sources.filter((source) => adopted.has(evidenceKey(source))),
      }))
    : lanes;
  if (!visibleLanes.length) return null;
  return (
    <section aria-label="Search lanes" className="rounded-lg border border-border bg-bg-subtle/60 p-4">
      <h2 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
        <Search className="h-3.5 w-3.5" aria-hidden="true" />
        Search lanes
        <span className="text-text-faint">({visibleLanes.length})</span>
      </h2>
      <div className="space-y-2">
        {visibleLanes.map((lane, i) => (
          <LaneRow key={`${lane.query}-${i}`} lane={lane} index={i} />
        ))}
      </div>
    </section>
  );
}
