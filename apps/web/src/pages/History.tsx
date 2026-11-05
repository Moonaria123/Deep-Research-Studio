import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Archive, CirclePause, FileText, Pin, PinOff, Search, Trash2 } from "lucide-react";
import type { Session } from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { formatTime } from "@/lib/utils";
import { useGsapEntrance } from "@/hooks/useGsapEntrance";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { toast } from "@/components/ui/toast";
import { ModeBadge } from "@/components/ModeBadge";
import { Badge } from "@/components/ui/badge";
import { IS_DEMO } from "@/lib/runtime";

type PendingAction = { type: "archive" | "delete"; session: Session };
const PIN_LIMIT = 10;

export function History() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [q, setQ] = useState("");
  const [pending, setPending] = useState<PendingAction>();
  const ref = useGsapEntrance<HTMLDivElement>([sessions.length]);

  const load = () => api.listSessions().then((r) => setSessions(r.sessions)).catch(() => undefined);

  useEffect(() => {
    load();
  }, []);

  const filtered = sessions.filter((s) => (s.title || "").toLowerCase().includes(q.toLowerCase()));
  const pinned = filtered.filter((s) => s.pinned);
  const recent = filtered.filter((s) => !s.pinned);

  const togglePinned = async (session: Session) => {
    const nextPinned = !session.pinned;
    if (nextPinned && sessions.filter((item) => item.pinned).length >= PIN_LIMIT) {
      toast.error(`Pinned sessions are limited to ${PIN_LIMIT}.`);
      return;
    }
    try {
      const result = await api.patchSession(session.id, { pinned: nextPinned });
      setSessions((prev) =>
        prev
          .map((item) => (item.id === session.id ? { ...item, ...result.session } : item))
          .sort((a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) || (b.updated_at ?? 0) - (a.updated_at ?? 0)),
      );
      toast.success(nextPinned ? "Session pinned." : "Session unpinned.");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update session pin.");
    }
  };

  const confirmAction = async () => {
    if (!pending) return;
    const { type, session } = pending;
    try {
      if (type === "archive") {
        await api.archiveSession(session.id);
        setSessions((prev) => prev.filter((s) => s.id !== session.id));
        toast.success("Session archived.");
      } else {
        await api.deleteSession(session.id);
        setSessions((prev) => prev.filter((s) => s.id !== session.id));
        toast.success("Session deleted.");
      }
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Action failed.");
    } finally {
      setPending(undefined);
    }
  };

  return (
    <div ref={ref} className="mx-auto w-full max-w-[820px] px-6 py-8">
      <div data-anim className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold text-text">History</h1>
        <Link to="/history/archived" className="text-sm text-accent hover:underline focus-ring rounded">
          已归档会话 →
        </Link>
      </div>
      <div data-anim className="relative mb-4">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-faint" aria-hidden />
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search history…"
          aria-label="Search history"
          className="pl-9"
        />
      </div>
      <div className="space-y-4">
        {filtered.length === 0 && (
          <div className="rounded-md border border-border bg-surface px-4 py-6 text-center text-sm text-text-muted">
            Nothing here yet.
          </div>
        )}
        {pinned.length > 0 && (
          <SessionList
            title="Pinned"
            subtitle={`${pinned.length}/${PIN_LIMIT} · stored locally`}
            sessions={pinned}
            onOpen={(id) => navigate(`/session/${id}`)}
            onTogglePinned={(session) => void togglePinned(session)}
            onArchive={(session) => setPending({ type: "archive", session })}
            onDelete={(session) => setPending({ type: "delete", session })}
          />
        )}
        {recent.length > 0 && (
          <SessionList
            title="Recent"
            sessions={recent}
            onOpen={(id) => navigate(`/session/${id}`)}
            onTogglePinned={(session) => void togglePinned(session)}
            onArchive={(session) => setPending({ type: "archive", session })}
            onDelete={(session) => setPending({ type: "delete", session })}
          />
        )}
      </div>

      <Dialog
        open={Boolean(pending)}
        onOpenChange={(open) => !open && setPending(undefined)}
        title={pending?.type === "delete" ? "Delete session" : "Archive session"}
        description={
          pending?.type === "delete"
            ? `Permanently delete "${pending.session.title || "Untitled"}" and all messages, runs, session memories, and its workspace session summary? This cannot be undone.`
            : pending
              ? `Archive "${pending.session.title || "Untitled"}"? It will be hidden from History and content will be locked until restored. Workspace session summaries are kept.`
              : ""
        }
      >
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" type="button" onClick={() => setPending(undefined)}>
            Cancel
          </Button>
          <Button
            variant={pending?.type === "delete" ? "danger" : "default"}
            size="sm"
            type="button"
            onClick={() => void confirmAction()}
            disabled={IS_DEMO}
          >
            {pending?.type === "delete" ? "Delete" : "Archive"}
          </Button>
        </div>
      </Dialog>
    </div>
  );
}

function SessionList({
  title,
  subtitle,
  sessions,
  onOpen,
  onTogglePinned,
  onArchive,
  onDelete,
}: {
  title: string;
  subtitle?: string;
  sessions: Session[];
  onOpen: (sessionId: string) => void;
  onTogglePinned: (session: Session) => void;
  onArchive: (session: Session) => void;
  onDelete: (session: Session) => void;
}) {
  return (
    <section>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-text">{title}</h2>
        {subtitle ? <span className="text-xs text-text-faint">{subtitle}</span> : null}
      </div>
      <ul className="divide-y divide-border rounded-md border border-border bg-surface">
        {sessions.map((s) => (
          <li key={s.id} data-anim className="flex items-center gap-1">
            <button
              className="flex min-w-0 flex-1 items-center gap-3 px-4 py-3 text-left hover:bg-bg-subtle focus-ring"
              onClick={() => onOpen(s.id)}
            >
              <FileText className="h-4 w-4 shrink-0 text-text-muted" aria-hidden />
              <span className="min-w-0 flex-1 truncate text-sm text-text-body">{s.title || "Untitled"}</span>
              <ModeBadge mode={s.latest_run_mode} />
              {s.latest_run_status === "paused" ? (
                <Badge variant={s.resumable_checkpoint?.resume_readiness === "degraded" ? "warning" : "accent"}>
                  <CirclePause className="h-3 w-3" /> Resume
                </Badge>
              ) : null}
              <span className="shrink-0 text-xs text-text-faint">{formatTime(s.updated_at)}</span>
            </button>
            <div className="flex shrink-0 items-center gap-1 pr-2">
              <Button
                variant="ghost"
                size="sm"
                type="button"
                aria-label={`${s.pinned ? "Unpin" : "Pin"} ${s.title || "session"}`}
                disabled={IS_DEMO}
                onClick={(e) => {
                  e.stopPropagation();
                  onTogglePinned(s);
                }}
              >
                {s.pinned ? <PinOff className="h-4 w-4" /> : <Pin className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                type="button"
                aria-label={`Archive ${s.title || "session"}`}
                disabled={IS_DEMO}
                onClick={(e) => {
                  e.stopPropagation();
                  onArchive(s);
                }}
              >
                <Archive className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                type="button"
                aria-label={`Delete ${s.title || "session"}`}
                disabled={IS_DEMO}
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(s);
                }}
              >
                <Trash2 className="h-4 w-4 text-danger" />
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
