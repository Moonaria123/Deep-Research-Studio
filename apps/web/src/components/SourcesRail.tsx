import type { EvidenceSource } from "@/lib/types";
import { evidenceKey } from "@/lib/evidenceRoutes";
import { SourceCard } from "./SourceCard";

export function SourcesRail({
  sources,
  highlightIndexes,
  onSourceFocus,
  onSourcePin,
}: {
  sources: EvidenceSource[];
  highlightIndexes?: number[];
  onSourceFocus?: (index: number | null) => void;
  onSourcePin?: (index: number) => void;
}) {
  const active = new Set(highlightIndexes ?? []);
  return (
    <aside aria-label="Sources" className="w-[300px] shrink-0 space-y-3">
      <h2 className="px-1 text-xs font-semibold uppercase tracking-wide text-text-faint">
        Sources · {sources.length}
      </h2>
      {sources.length === 0 ? (
        <p className="px-1 text-sm text-text-muted">No sources for this run.</p>
      ) : (
        sources.map((source, i) => (
          <SourceCard
            key={evidenceKey(source)}
            index={i + 1}
            source={source}
            active={active.has(i + 1)}
            onFocus={onSourceFocus}
            onPin={onSourcePin}
          />
        ))
      )}
    </aside>
  );
}
