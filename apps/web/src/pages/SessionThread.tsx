import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { ArrowLeft, FileText, History, Zap, Layers, Telescope, Users } from "lucide-react";
import type { Message, Mode, SearchUsagePolicy } from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { isMode } from "@/lib/mode";
import { useGsapEntrance } from "@/hooks/useGsapEntrance";
import { useSplitPaneRatio } from "@/hooks/useSplitPaneRatio";
import { Composer, type ComposerOptions } from "@/components/Composer";
import { ModeBadge } from "@/components/ModeBadge";
import { ModeCard, type ModeOption } from "@/components/ModeCard";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/toast";
import { RunPanel } from "@/pages/RunView";
import { IS_DEMO } from "@/lib/runtime";

const MODES: ModeOption[] = [
  { mode: "fast", title: "Fast Search", subtitle: "One quick retrieval pass for a concise grounded answer.", icon: Zap },
  { mode: "multiple", title: "Multiple Search", subtitle: "Fan out parallel sub-queries, then merge with RRF.", icon: Layers },
  { mode: "deep", title: "Deep Research", subtitle: "Plan, search, analyze, and write a sourced report.", icon: Telescope },
  { mode: "council", title: "Models Council", subtitle: "Several models debate shared evidence for consensus.", icon: Users },
];

/** Parse the JSON-encoded message metadata, tolerating malformed values. */
function parseMeta(message: Message): { run_id?: string; mode?: Mode } {
  if (!message.metadata) return {};
  try {
    const meta = JSON.parse(message.metadata) as Record<string, unknown>;
    const runId = typeof meta.run_id === "string" ? meta.run_id : undefined;
    const mode = isMode(meta.mode as string) ? (meta.mode as Mode) : undefined;
    return { run_id: runId, mode };
  } catch {
    return {};
  }
}

const PREVIEW_CHARS = 420;

function previewOf(text: string): { body: string; truncated: boolean } {
  const trimmed = text.trim();
  if (trimmed.length <= PREVIEW_CHARS) return { body: trimmed, truncated: false };
  return { body: `${trimmed.slice(0, PREVIEW_CHARS).trimEnd()}…`, truncated: true };
}

/** Threaded view of a single session: prior turns plus a follow-up composer. */
export function SessionThread() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedRunId = searchParams.get("run") || undefined;
  const [messages, setMessages] = useState<Message[]>([]);
  const [title, setTitle] = useState("Conversation");
  const [mode, setMode] = useState<Mode>("fast");
  const [modeDefault, setModeDefault] = useState<Mode | null>(null);
  const [searchUsagePolicy, setSearchUsagePolicy] = useState<SearchUsagePolicy | null>(null);
  const [settingDefault, setSettingDefault] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | undefined>(selectedRunId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [archived, setArchived] = useState(false);
  // DR-ITER-053: live semantic recall of earlier same-session turns for the draft query.
  const [draft, setDraft] = useState("");
  const [related, setRelated] = useState<Message[]>([]);
  const ref = useGsapEntrance<HTMLDivElement>([messages.length]);
  const splitPane = useSplitPaneRatio(sessionId);

  useEffect(() => {
    if (!sessionId) return;
    let disposed = false;
    Promise.all([api.listMessages(sessionId), api.listSessions()])
      .then(([messageResult, sessionResult]) => {
        if (disposed) return;
        setMessages(messageResult.messages);
        setArchived(false);
        const found = sessionResult.sessions.find((s) => s.id === sessionId);
        if (found?.title) setTitle(found.title);
        const sessionDefault = found?.mode_default ?? null;
        const lastMode = [...messageResult.messages].reverse().map(parseMeta).find((m) => m.mode)?.mode;
        setModeDefault(sessionDefault);
        setSearchUsagePolicy(found?.search_usage_policy ?? null);
        if (sessionDefault) setMode(sessionDefault);
        else if (lastMode) setMode(lastMode);
        if (!selectedRunId) {
          const latestMessageRun = [...messageResult.messages]
            .reverse()
            .map(parseMeta)
            .find((m) => m.run_id)?.run_id;
          setActiveRunId(latestMessageRun ?? found?.latest_run_id ?? undefined);
        }
      })
      .catch((e) => {
        if (disposed) return;
        if (e instanceof ApiError && e.code === "SESSION_ARCHIVED") setArchived(true);
        else setError(e instanceof ApiError ? e.message : (e as Error).message);
      });
    return () => {
      disposed = true;
    };
  }, [sessionId, selectedRunId]);

  useEffect(() => {
    setActiveRunId(selectedRunId);
  }, [selectedRunId]);

  const turns = useMemo(
    () => messages.filter((m) => m.role === "user" || m.role === "assistant"),
    [messages],
  );

  // Debounce the draft query, then recall the most relevant earlier turns (intra
  // scope). Empty results / no embedder simply leave the panel hidden.
  useEffect(() => {
    if (!sessionId || archived) return;
    const q = draft.trim();
    if (q.length < 3) {
      setRelated([]);
      return;
    }
    const timer = setTimeout(() => {
      api
        .searchHistory(sessionId, { query: q, scope: "intra", limit: 5 })
        .then((r) => setRelated(r.results))
        .catch(() => setRelated([]));
    }, 300);
    return () => clearTimeout(timer);
  }, [draft, sessionId, archived]);

  const start = async (query: string, options?: ComposerOptions) => {
    if (!sessionId) return;
    setBusy(true);
    setError(undefined);
    try {
      const { run_id } = await api.createRun({
        query,
        mode,
        session_id: sessionId,
        budget: options?.budget,
        llm_model: options?.llm_model,
        council_models: options?.council_models,
        council_size: options?.council_size,
        council_strategy: options?.council_strategy,
        temperature: options?.temperature,
        thinking_effort: options?.thinking_effort,
        memory_policy: options?.memory_policy,
        library_scope: options?.library_scope,
        context_budget: options?.context_budget,
        search_options: options?.search_options,
        search_usage_policy: options?.search_usage_policy,
      });
      setActiveRunId(run_id);
      setSearchParams((current) => {
        const next = new URLSearchParams(current);
        next.set("run", run_id);
        return next;
      });
      setBusy(false);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
      setBusy(false);
    }
  };

  const openRunInPanel = (runId: string) => {
    setActiveRunId(runId);
    setSearchParams((current) => {
      const next = new URLSearchParams(current);
      next.set("run", runId);
      return next;
    });
  };

  const saveModeDefault = async () => {
    if (!sessionId || settingDefault || modeDefault === mode) return;
    setSettingDefault(true);
    try {
      const result = await api.patchSession(sessionId, { mode_default: mode });
      setModeDefault(result.session.mode_default ?? mode);
      toast.success("Session default mode saved.");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not save session default mode.");
    } finally {
      setSettingDefault(false);
    }
  };

  const saveSearchUsagePolicy = async (policy: SearchUsagePolicy | null) => {
    if (!sessionId) throw new Error("Session is unavailable.");
    const result = await api.patchSession(sessionId, { search_usage_policy: policy });
    setSearchUsagePolicy(result.session.search_usage_policy ?? null);
    toast.success(policy ? "Session search strategy saved." : "Session now follows the global search strategy.");
  };

  return (
    <div ref={ref} className="mx-auto w-full max-w-[1440px] px-4 py-6 sm:px-6 lg:py-8">
      <div data-anim className="mb-4 flex items-center justify-between gap-3">
        <button
          className="inline-flex items-center gap-1 text-sm text-text-muted hover:text-text focus-ring rounded"
          onClick={() => navigate("/history")}
          aria-label="Back to history"
        >
          <ArrowLeft className="h-4 w-4" /> History
        </button>
      </div>
      <h1 data-anim className="mb-4 text-2xl font-bold text-text">
        {title}
      </h1>

      {archived ? (
        <Card data-anim className="border-danger/40">
          <CardContent className="space-y-3 pt-5">
            <p role="alert" className="text-sm text-danger">
              该会话已归档，内容已锁定。请先在已归档会话中恢复后再继续提问。
            </p>
            <Link
              to="/history/archived"
              className="inline-flex h-8 items-center rounded-md border border-border bg-surface px-3 text-sm text-text hover:bg-bg-subtle focus-ring"
            >
              前往已归档会话
            </Link>
          </CardContent>
        </Card>
      ) : (
        <div
          ref={splitPane.containerRef}
          data-testid="session-split-layout"
          className="grid min-w-0 grid-cols-1 gap-6 lg:flex lg:items-stretch lg:gap-0"
        >
          <section
            className="min-w-0 space-y-3 lg:min-w-[280px] lg:shrink"
            style={{ flexBasis: `${splitPane.ratio * 100}%` }}
          >
            {turns.length === 0 && (
              <p data-anim className="text-sm text-text-muted">
                No messages yet. Ask a question below to start the conversation.
              </p>
            )}
            {turns.map((m) => {
              if (m.role === "user") {
                return (
                  <div key={m.id} data-anim className="flex justify-end">
                    <div className="max-w-[85%] rounded-lg bg-accent px-4 py-2 text-sm text-accent-contrast">
                      {m.content}
                    </div>
                  </div>
                );
              }
              const meta = parseMeta(m);
              const { body, truncated } = previewOf(m.content);
              return (
                <Card key={m.id} data-anim>
                  <CardContent className="space-y-2 pt-4">
                    <div className="flex items-center gap-2">
                      <ModeBadge mode={meta.mode} />
                    </div>
                    <p className="whitespace-pre-wrap text-[15px] leading-7 text-text-body">{body}</p>
                    {meta.run_id && (
                      <button
                        type="button"
                        onClick={() => openRunInPanel(meta.run_id!)}
                        className="inline-flex items-center gap-1 rounded text-sm text-accent hover:underline focus-ring"
                      >
                        <FileText className="h-4 w-4" /> {truncated ? "查看完整报告" : "查看运行详情"}
                      </button>
                    )}
                  </CardContent>
                </Card>
              );
            })}

            <div className="mt-6 space-y-3">
            {related.length > 0 && (
              <Card data-anim data-testid="related-turns" className="border-accent/30">
                <CardContent className="space-y-2 pt-4">
                  <div className="flex items-center gap-2 text-sm font-medium text-text">
                    <History className="h-4 w-4 text-accent" /> Related earlier turns
                  </div>
                  <ul className="space-y-2">
                    {related.map((m) => {
                      const meta = parseMeta(m);
                      const { body } = previewOf(m.content);
                      return (
                        <li key={m.id} className="text-sm text-text-body">
                          <span className="mr-1 text-text-faint">
                            {m.role === "user" ? "You:" : "Answer:"}
                          </span>
                          {meta.run_id ? (
                            <button
                              type="button"
                              onClick={() => openRunInPanel(meta.run_id!)}
                              className="rounded text-left text-text-body hover:underline focus-ring"
                            >
                              {body}
                            </button>
                          ) : (
                            <span>{body}</span>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </CardContent>
              </Card>
            )}
            <Composer
              onSubmit={start}
              busy={busy}
              mode={mode}
              placeholder="Ask a follow-up…"
              onQueryChange={setDraft}
              defaultUseMemory
              initialSearchUsagePolicy={searchUsagePolicy}
              onSearchUsagePolicyChange={saveSearchUsagePolicy}
            />
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-bg-subtle px-3 py-2 text-xs text-text-muted">
              <span>
                Session default: <span className="font-medium text-text-body">{modeDefault ?? "not set"}</span>
              </span>
              <Button
                type="button"
                variant={modeDefault === mode ? "ghost" : "outline"}
                size="sm"
                disabled={IS_DEMO || settingDefault || modeDefault === mode}
                onClick={() => void saveModeDefault()}
              >
                {modeDefault === mode ? "This is the default" : "Set as session default"}
              </Button>
            </div>
            {error && (
              <p role="alert" className="text-sm text-danger">
                {error}
              </p>
            )}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {MODES.map((option) => (
                <ModeCard key={option.mode} option={option} active={mode === option.mode} onSelect={setMode} />
              ))}
            </div>
          </div>
          </section>

          <div
            data-testid="session-split-handle"
            aria-label="Resize session split pane"
            className="hidden w-3 shrink-0 cursor-col-resize items-stretch justify-center rounded-sm focus-ring lg:flex"
            style={{ touchAction: "none" }}
            {...splitPane.separatorProps}
          >
            <span className="my-1 w-px rounded-full bg-border transition-colors hover:bg-accent" aria-hidden />
          </div>

          <aside className="min-w-0 rounded-md border border-border bg-surface/70 p-3 shadow-sm lg:min-w-[280px] lg:flex-1">
            <RunPanel
              runId={activeRunId}
              embedded
              onFollowUp={() => undefined}
              onChildRun={openRunInPanel}
            />
          </aside>
        </div>
      )}
    </div>
  );
}
