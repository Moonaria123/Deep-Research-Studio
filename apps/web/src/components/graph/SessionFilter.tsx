import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { SessionFilterMode, SessionFilterOption } from "@/lib/graphSessionFilter";

export function SessionFilter({
  sessions,
  mode,
  selectedIds,
  onToggleSession,
  onShowAll,
  onShowCrossLinked,
  onClear,
  onResetDefault,
}: {
  sessions: SessionFilterOption[];
  mode: SessionFilterMode;
  selectedIds: Set<string>;
  onToggleSession: (sessionId: string) => void;
  onShowAll: () => void;
  onShowCrossLinked: () => void;
  onClear: () => void;
  onResetDefault: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return sessions;
    return sessions.filter((session) => [session.title, session.id].some((value) => value.toLowerCase().includes(needle)));
  }, [query, sessions]);
  const modeLabel = mode === "all" ? "All" : mode === "manual" ? "Manual" : mode === "cross_linked" ? "With links" : "Default";

  useEffect(() => {
    if (!open) return undefined;
    const handleClick = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="min-w-[160px] max-w-full justify-between"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((current) => !current)}
      >
        <span>Sessions · {selectedIds.size} / {sessions.length}</span>
        <ChevronDown className="h-4 w-4 text-text-faint" aria-hidden />
      </Button>
      {open ? (
        <div className="absolute left-0 top-full z-20 mt-2 w-[320px] max-w-[calc(100vw_-_2rem)] rounded-md border border-border bg-bg p-3 shadow-lg">
          <div className="mb-3 flex items-center justify-between gap-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-text-muted">{modeLabel}</span>
            <div className="flex gap-1">
              <Button type="button" variant="ghost" size="sm" onClick={onResetDefault}>
                Default
              </Button>
              <Button type="button" variant="ghost" size="sm" onClick={onShowAll}>
                Show all
              </Button>
              <Button type="button" variant="ghost" size="sm" onClick={onShowCrossLinked}>
                With links
              </Button>
              <Button type="button" variant="ghost" size="sm" onClick={onClear}>
                Clear
              </Button>
            </div>
          </div>
          <label className="relative mb-3 block">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-faint" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="pl-9"
              placeholder="Search sessions"
              aria-label="Search sessions"
            />
          </label>
          <div className="max-h-64 overflow-y-auto pr-1" role="group" aria-label="Session choices">
            {filtered.length ? (
              filtered.map((session) => {
                const checked = selectedIds.has(session.id);
                return (
                  <label
                    key={session.id}
                    className={cn(
                      "flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors",
                      checked ? "bg-accent/10 text-text" : "text-text-muted hover:bg-bg-subtle hover:text-text",
                    )}
                  >
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded border-border text-accent"
                      checked={checked}
                      onChange={() => onToggleSession(session.id)}
                    />
                    <span className="min-w-0 flex-1 truncate">{session.title}</span>
                  </label>
                );
              })
            ) : (
              <p className="px-2 py-6 text-center text-sm text-text-muted">No matching sessions</p>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
