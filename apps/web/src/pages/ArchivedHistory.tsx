import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArchiveRestore, FileText, Trash2 } from "lucide-react";
import type { ArchivedSession } from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { formatTime } from "@/lib/utils";
import { useGsapEntrance } from "@/hooks/useGsapEntrance";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { toast } from "@/components/ui/toast";
import { IS_DEMO } from "@/lib/runtime";
import { ModeBadge } from "@/components/ModeBadge";

export function ArchivedHistory() {
  const [sessions, setSessions] = useState<ArchivedSession[]>([]);
  const [deleting, setDeleting] = useState<ArchivedSession>();
  const ref = useGsapEntrance<HTMLDivElement>([sessions.length]);

  const load = () =>
    api.listArchivedSessions().then((r) => setSessions(r.sessions)).catch(() => undefined);

  useEffect(() => {
    load();
  }, []);

  const restore = async (session: ArchivedSession) => {
    try {
      await api.restoreSession(session.id);
      setSessions((prev) => prev.filter((s) => s.id !== session.id));
      toast.success("Session restored.");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not restore session.");
    }
  };

  const confirmDelete = async () => {
    if (!deleting) return;
    const id = deleting.id;
    try {
      await api.deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      toast.success("Session deleted.");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not delete session.");
    } finally {
      setDeleting(undefined);
    }
  };

  return (
    <div ref={ref} className="mx-auto w-full max-w-[820px] px-6 py-8">
      <div data-anim className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold text-text">已归档会话</h1>
        <Link to="/history" className="text-sm text-accent hover:underline focus-ring rounded">
          ← Back to History
        </Link>
      </div>
      <ul className="divide-y divide-border rounded-md border border-border bg-surface">
        {sessions.length === 0 && (
          <li className="px-4 py-6 text-center text-sm text-text-muted">暂无已归档会话</li>
        )}
        {sessions.map((s) => (
          <li key={s.id} data-anim className="flex items-center gap-3 px-4 py-3">
            <FileText className="h-4 w-4 shrink-0 text-text-muted" aria-hidden />
            <div className="min-w-0 flex-1">
              <div className="flex min-w-0 items-center gap-2">
                <p className="min-w-0 flex-1 truncate text-sm text-text-body">{s.title || "Untitled"}</p>
                <ModeBadge mode={s.latest_run_mode} />
              </div>
              <p className="text-xs text-text-faint">Archived {formatTime(s.archived_at)}</p>
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <Button
                variant="outline"
                size="sm"
                type="button"
                onClick={() => void restore(s)}
                disabled={IS_DEMO}
              >
                <ArchiveRestore className="h-4 w-4" /> Restore
              </Button>
              <Button
                variant="ghost"
                size="sm"
                type="button"
                aria-label={`Delete ${s.title || "session"}`}
                onClick={() => setDeleting(s)}
                disabled={IS_DEMO}
              >
                <Trash2 className="h-4 w-4 text-danger" />
              </Button>
            </div>
          </li>
        ))}
      </ul>

      <Dialog
        open={Boolean(deleting)}
        onOpenChange={(open) => !open && setDeleting(undefined)}
        title="Delete session"
        description={
          deleting
            ? `Permanently delete "${deleting.title || "Untitled"}" and all messages, runs, session memories, and its workspace session summary? This cannot be undone.`
            : ""
        }
      >
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" type="button" onClick={() => setDeleting(undefined)}>
            Cancel
          </Button>
          <Button variant="danger" size="sm" type="button" onClick={() => void confirmDelete()} disabled={IS_DEMO}>
            Delete
          </Button>
        </div>
      </Dialog>
    </div>
  );
}
