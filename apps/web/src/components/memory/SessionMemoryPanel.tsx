import { useCallback, useEffect, useMemo, useState } from "react";
import { Archive, ExternalLink, RotateCcw, Search, Trash2 } from "lucide-react";
import { Link, useInRouterContext } from "react-router-dom";
import type { MemoryItem, SessionMemoryGroup, SessionMemoryList } from "@/lib/types";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { toast } from "@/components/ui/toast";
import { IS_DEMO } from "@/lib/runtime";

const EMPTY_RESULT: SessionMemoryList = {
  sessions: [],
  total: 0,
  limit: 50,
  offset: 0,
  truncated: false,
};

function formatDate(value?: number | null) {
  if (!value) return "";
  const date = new Date(value * 1000);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString();
}

function OpenSessionLink({ sessionId }: { sessionId: string }) {
  const inRouter = useInRouterContext();
  const href = `/session/${encodeURIComponent(sessionId)}`;
  const className = "inline-flex h-8 items-center gap-1 rounded-sm px-2 text-xs font-medium text-text-body hover:bg-bg-subtle focus-ring";
  const content = <><ExternalLink className="h-3.5 w-3.5" /> Open Session</>;
  return inRouter ? <Link to={href} className={className}>{content}</Link> : <a href={href} className={className}>{content}</a>;
}

export function SessionMemoryPanel() {
  const [result, setResult] = useState<SessionMemoryList>(EMPTY_RESULT);
  const [queryDraft, setQueryDraft] = useState("");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busyId, setBusyId] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<MemoryItem | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const next = await api.listSessionMemories({
        workspace_key: "default",
        query,
        include_archived_sessions: true,
        include_archived_memories: true,
        limit: 50,
        offset: 0,
      });
      setResult(next);
    } catch {
      setError("Session memories could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, [query]);

  useEffect(() => {
    void load();
  }, [load]);

  const activeGroups = useMemo(
    () => result.sessions.filter((group) => !group.session_archived_at),
    [result.sessions],
  );
  const archivedGroups = useMemo(
    () => result.sessions.filter((group) => Boolean(group.session_archived_at)),
    [result.sessions],
  );

  const runAction = async (memory: MemoryItem, action: "archive" | "restore" | "delete") => {
    if (busyId) return;
    setBusyId(memory.id);
    try {
      if (action === "archive") await api.archiveMemory(memory.id);
      else if (action === "restore") await api.restoreMemory(memory.id);
      else await api.deleteMemory(memory.id);
      toast.success(
        action === "archive" ? "Archived Session memory." : action === "restore" ? "Restored Session memory." : "Deleted Session memory.",
      );
      setDeleteTarget(null);
      await load();
    } catch {
      toast.error(`Could not ${action} Session memory.`);
    } finally {
      setBusyId("");
    }
  };

  const renderGroups = (groups: SessionMemoryGroup[], heading: string) => {
    if (groups.length === 0) return null;
    return (
      <section aria-label={`${heading} Sessions`} className="space-y-3">
        <h3 className="text-xs font-semibold uppercase text-text-faint">{heading}</h3>
        {groups.map((group) => (
          <div key={group.session_id} className="border-b border-border pb-4 last:border-b-0">
            <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h4 className="break-words text-sm font-semibold text-text">{group.session_title || "Untitled session"}</h4>
                  {group.session_archived_at ? <Badge variant="default">Archived Session</Badge> : null}
                </div>
                <p className="mt-1 text-xs text-text-faint">
                  {group.memories.length} {group.memories.length === 1 ? "memory" : "memories"}
                  {formatDate(group.session_updated_at) ? ` · ${formatDate(group.session_updated_at)}` : ""}
                </p>
              </div>
              <OpenSessionLink sessionId={group.session_id} />
            </div>
            <ul className="space-y-2">
              {group.memories.map((memory) => (
                <li key={memory.id} className="rounded-sm border border-border bg-bg-subtle px-3 py-2">
                  <p className="line-clamp-4 break-words text-sm text-text-body" title={memory.content}>
                    {memory.content}
                  </p>
                  <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-text-faint">
                      <Badge variant="accent">Session</Badge>
                      {memory.archived_at ? <span>Archived {formatDate(memory.archived_at)}</span> : null}
                    </div>
                    <div className="flex items-center gap-1">
                      {memory.archived_at ? (
                        <Button
                          type="button"
                          size="icon"
                          variant="ghost"
                          disabled={IS_DEMO || busyId === memory.id}
                          onClick={() => void runAction(memory, "restore")}
                          aria-label={`Restore Session memory ${memory.id}`}
                          title="Restore"
                        >
                          <RotateCcw className="h-4 w-4" />
                        </Button>
                      ) : (
                        <Button
                          type="button"
                          size="icon"
                          variant="ghost"
                          disabled={IS_DEMO || busyId === memory.id}
                          onClick={() => void runAction(memory, "archive")}
                          aria-label={`Archive Session memory ${memory.id}`}
                          title="Archive"
                        >
                          <Archive className="h-4 w-4" />
                        </Button>
                      )}
                      <Button
                        type="button"
                        size="icon"
                        variant="ghost"
                        disabled={IS_DEMO || busyId === memory.id}
                        onClick={() => setDeleteTarget(memory)}
                        aria-label={`Delete Session memory ${memory.id}`}
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </section>
    );
  };

  return (
    <section className="mb-5 border-t border-border pt-5" aria-labelledby="session-memory-heading">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="session-memory-heading" className="text-sm font-semibold text-text">Session memories</h2>
          <p className="mt-1 text-xs text-text-faint">{result.total} Sessions</p>
        </div>
        <form
          role="search"
          className="flex w-full max-w-sm gap-2 sm:w-auto"
          onSubmit={(event) => {
            event.preventDefault();
            const next = queryDraft.trim();
            if (next === query) void load();
            else setQuery(next);
          }}
        >
          <Input
            type="search"
            value={queryDraft}
            onChange={(event) => setQueryDraft(event.target.value)}
            aria-label="Search Session memories"
            placeholder="Search Sessions"
            className="min-w-0"
          />
          <Button type="submit" size="icon" variant="outline" aria-label="Search Session memories now" title="Search">
            <Search className="h-4 w-4" />
          </Button>
        </form>
      </div>

      {error ? <p role="alert" className="mb-3 text-sm text-danger">{error}</p> : null}
      {loading && result.sessions.length === 0 ? <p className="text-sm text-text-muted">Loading Session memories...</p> : null}
      {!loading && !error && result.sessions.length === 0 ? (
        <p className="text-sm text-text-muted">No Session memories found.</p>
      ) : null}
      <div className="space-y-5">
        {renderGroups(activeGroups, "Active")}
        {renderGroups(archivedGroups, "Archived")}
      </div>

      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
        title="Delete Session memory"
        description="This permanently removes the selected Session memory."
      >
        {deleteTarget ? (
          <div className="space-y-4">
            <p className="break-words text-sm text-text-body">{deleteTarget.content}</p>
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={() => setDeleteTarget(null)}>Cancel</Button>
              <Button
                type="button"
                variant="danger"
                disabled={IS_DEMO || busyId === deleteTarget.id}
                onClick={() => void runAction(deleteTarget, "delete")}
              >
                Delete Session memory
              </Button>
            </div>
          </div>
        ) : null}
      </Dialog>
    </section>
  );
}
