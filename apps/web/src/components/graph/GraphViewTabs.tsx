import { cn } from "@/lib/utils";

export type KnowledgeGraphViewMode = "session_tree" | "wiki_map";

const TABS: { id: KnowledgeGraphViewMode; label: string }[] = [
  { id: "session_tree", label: "Session Map" },
  { id: "wiki_map", label: "Wiki Map" },
];

export function GraphViewTabs({
  value,
  onChange,
}: {
  value: KnowledgeGraphViewMode;
  onChange: (value: KnowledgeGraphViewMode) => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-border bg-bg p-1" aria-label="Graph view mode">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          aria-pressed={value === tab.id}
          onClick={() => onChange(tab.id)}
          className={cn(
            "rounded-sm px-3 py-1.5 text-xs font-semibold transition-colors focus-ring",
            value === tab.id ? "bg-accent text-white" : "text-text-muted hover:bg-bg-subtle hover:text-text",
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
