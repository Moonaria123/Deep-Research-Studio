import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Zap, Layers, Telescope, Users, FileText, ChevronRight } from "lucide-react";
import type { Mode, Session } from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { formatTime } from "@/lib/utils";
import { useProvider } from "@/components/AppShell";
import { useGsapEntrance } from "@/hooks/useGsapEntrance";
import { Composer, type ComposerOptions } from "@/components/Composer";
import { ConnectorStrip } from "@/components/ConnectorStrip";
import { ModeCard, type ModeOption } from "@/components/ModeCard";
import { ModeBadge } from "@/components/ModeBadge";

const MODES: ModeOption[] = [
  { mode: "fast", title: "Fast Search", subtitle: "One quick retrieval pass for a concise grounded answer.", icon: Zap },
  { mode: "multiple", title: "Multiple Search", subtitle: "Fan out parallel sub-queries, then merge with RRF.", icon: Layers },
  { mode: "deep", title: "Deep Research", subtitle: "Plan, search, analyze, and write a sourced report.", icon: Telescope },
  { mode: "council", title: "Models Council", subtitle: "Several models debate shared evidence for consensus.", icon: Users },
];

export function Home() {
  const navigate = useNavigate();
  const provider = useProvider();
  const [mode, setMode] = useState<Mode>("fast");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [recent, setRecent] = useState<Session[]>([]);
  const ref = useGsapEntrance<HTMLDivElement>([]);

  useEffect(() => {
    api.listSessions().then((r) => setRecent(r.sessions.slice(0, 5))).catch(() => undefined);
  }, []);

  const start = async (query: string, options?: ComposerOptions) => {
    setBusy(true);
    setError(undefined);
    try {
      const { run_id } = await api.createRun({
        query,
        mode,
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
      navigate(`/run/${run_id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
      setBusy(false);
    }
  };

  return (
    <div ref={ref} className="mx-auto w-full max-w-[720px] px-6 py-10">
      <div data-anim className="mb-6 text-center">
        <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-lg bg-accent text-accent-contrast text-xl font-bold">
          R
        </div>
        <h1 className="text-3xl font-bold text-text">Welcome to Deep Research Studio</h1>
        <p className="mt-1 text-text-muted">Local-first research with your own keys and models.</p>
      </div>

      <Composer onSubmit={start} busy={busy} mode={mode} />
      {error && (
        <p role="alert" className="mt-2 text-sm text-danger">
          {error}
        </p>
      )}

      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {MODES.map((option) => (
          <ModeCard key={option.mode} option={option} active={mode === option.mode} onSelect={setMode} />
        ))}
      </div>

      <div className="mt-6">
        <ConnectorStrip provider={provider} />
      </div>

      <div data-anim className="mt-8">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text">Recent</h2>
          <button className="text-xs text-accent hover:underline focus-ring" onClick={() => navigate("/history")}>
            View all
          </button>
        </div>
        <ul className="divide-y divide-border rounded-md border border-border bg-surface">
          {recent.length === 0 && <li className="px-4 py-3 text-sm text-text-muted">No research yet.</li>}
          {recent.map((s) => (
            <li key={s.id}>
              <button
                className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-bg-subtle focus-ring"
                onClick={() => navigate(`/session/${s.id}`)}
              >
                <FileText className="h-4 w-4 shrink-0 text-text-muted" aria-hidden />
                <span className="min-w-0 flex-1 truncate text-sm text-text-body">{s.title || "Untitled"}</span>
                <ModeBadge mode={s.latest_run_mode} />
                <span className="shrink-0 text-xs text-text-faint">{formatTime(s.updated_at)}</span>
                <ChevronRight className="h-4 w-4 text-text-faint" aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      </div>

      <p data-anim className="mt-8 text-center text-xs text-text-faint">
        All data and notes are stored locally on your device.
      </p>
    </div>
  );
}
