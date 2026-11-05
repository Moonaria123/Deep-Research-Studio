import { useEffect, useState } from "react";
import { CheckCircle2, CirclePause, XCircle, RefreshCw, Save, Plug, Server, Database, Lock, Boxes, Trash2, Power } from "lucide-react";
import type {
  DeepCheckpointSettings,
  ProviderStatus,
  ProviderUpdate,
  SearchProviderCapability,
  SearchProviderId,
  SearchUsagePolicy,
  ToolsTestResult,
} from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { useGsapEntrance } from "@/hooks/useGsapEntrance";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog } from "@/components/ui/dialog";
import { ProviderRegistry } from "@/components/ProviderRegistry";
import { ContextBudgetEditor } from "@/components/ContextBudgetEditor";
import { PostRunNotesEditor } from "@/components/PostRunNotesEditor";
import { KnowledgeGraphSettingsEditor } from "@/components/KnowledgeGraphSettingsEditor";
import { RetrievalSettingsEditor } from "@/components/RetrievalSettingsEditor";
import { SearchUsageEditor } from "@/components/SearchUsageEditor";
import { selectLatestProviderStatus, useProviderActions } from "@/components/AppShell";
import { IS_DEMO } from "@/lib/runtime";

function StatusPill({ ok, label }: { ok?: boolean; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-xs font-medium text-text-muted">
      {ok ? (
        <CheckCircle2 className="h-3.5 w-3.5 text-success" aria-label="ready" />
      ) : (
        <XCircle className="h-3.5 w-3.5 text-text-faint" aria-label="not configured" />
      )}
      {label}
    </span>
  );
}

function Field({
  label,
  hint,
  ...props
}: { label: string; hint?: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="block space-y-1.5">
      <span className="text-sm font-medium text-text">{label}</span>
      <Input {...props} />
      {hint && <span className="block text-xs text-text-faint">{hint}</span>}
    </label>
  );
}

const EMPTY: ProviderUpdate = {
  brave_api_key: "",
  firecrawl_api_key: "",
  firecrawl_base_url: "",
  exa_api_key: "",
  tavily_api_key: "",
};

type HealthInfo = Awaited<ReturnType<typeof api.health>>;

export function Settings() {
  const [status, setStatus] = useState<ProviderStatus>();
  const [health, setHealth] = useState<HealthInfo>();
  const [form, setForm] = useState<ProviderUpdate>(EMPTY);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ kind: "ok" | "warning" | "error"; text: string }>();
  const [toolsTest, setToolsTest] = useState<ToolsTestResult | { error: string }>();
  const [testingTools, setTestingTools] = useState(false);
  const [removeProviders, setRemoveProviders] = useState<SearchProviderId[]>([]);
  const [searchUsage, setSearchUsage] = useState<SearchUsagePolicy>();
  const [searchUsageSaving, setSearchUsageSaving] = useState(false);
  const [searchUsageMessage, setSearchUsageMessage] = useState<string>();
  const [deepCheckpoint, setDeepCheckpoint] = useState<DeepCheckpointSettings>();
  const [deepCheckpointSaving, setDeepCheckpointSaving] = useState(false);
  const [deepCheckpointMessage, setDeepCheckpointMessage] = useState<string>();
  const [cacheDialogOpen, setCacheDialogOpen] = useState(false);
  const [cacheClearing, setCacheClearing] = useState(false);
  const [cacheMessage, setCacheMessage] = useState<{ kind: "ok" | "error"; text: string }>();
  const { publishProviderStatus } = useProviderActions();
  const ref = useGsapEntrance<HTMLDivElement>([status]);

  const commitProviderStatus = (next: ProviderStatus) => {
    setStatus((current) => selectLatestProviderStatus(current, next));
    publishProviderStatus(next);
  };

  const testTools = async (provider: SearchProviderId) => {
    setTestingTools(true);
    setToolsTest(undefined);
    try {
      const result = await api.testTools({
        provider,
        brave_api_key: form.brave_api_key?.trim() || undefined,
        firecrawl_api_key: form.firecrawl_api_key?.trim() || undefined,
        firecrawl_base_url: form.firecrawl_base_url?.trim() || undefined,
        exa_api_key: form.exa_api_key?.trim() || undefined,
        tavily_api_key: form.tavily_api_key?.trim() || undefined,
      });
      setToolsTest(result);
    } catch (err) {
      setToolsTest({ error: err instanceof ApiError ? err.message : "test failed" });
    } finally {
      setTestingTools(false);
    }
  };

  const load = () => {
    api
      .providers()
      .then(commitProviderStatus)
      .catch(() => undefined);
    api
      .health()
      .then((next) => setHealth(next))
      .catch(() => undefined);
    api
      .getSearchUsage()
      .then(setSearchUsage)
      .catch(() => undefined);
    api
      .getDeepCheckpointSettings()
      .then(setDeepCheckpoint)
      .catch(() => undefined);
  };

  useEffect(() => {
    load();
  }, []);

  const update = (key: keyof ProviderUpdate) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((prev) => ({ ...prev, [key]: e.target.value }));

  const saveSearchProviders = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setMessage(undefined);
    const providers: Parameters<typeof api.saveSearchProviders>[0]["providers"] = {};
    const drafts: Array<[SearchProviderId, keyof ProviderUpdate]> = [
      ["brave", "brave_api_key"],
      ["tavily", "tavily_api_key"],
      ["exa", "exa_api_key"],
      ["firecrawl", "firecrawl_api_key"],
    ];
    drafts.forEach(([provider, field]) => {
      const apiKey = (form[field] ?? "").trim();
      const remove = removeProviders.includes(provider);
      if (apiKey || remove) providers[provider] = { api_key: apiKey || undefined, remove };
    });
    const firecrawlBase = (form.firecrawl_base_url ?? "").trim();
    if (firecrawlBase) providers.firecrawl = { ...(providers.firecrawl ?? {}), base_url: firecrawlBase };
    try {
      const next = await api.saveSearchProviders({
        expected_revision: status?.provider_config_revision ?? 0,
        providers,
      });
      const nextStatus = status ? {
        ...status,
        provider_config_revision: next.revision,
        search_providers: next.providers,
        external_search_service: next.external_search_service,
        brave_ready: next.providers.brave.configured,
        tavily_ready: next.providers.tavily.configured,
        exa_ready: next.providers.exa.configured,
        firecrawl_ready: next.providers.firecrawl.configured,
      } : undefined;
      if (nextStatus) commitProviderStatus(nextStatus);
      else api.providers().then(commitProviderStatus).catch(() => undefined);
      const accepted = new Set(next.saved ?? []);
      setForm((prev) => ({
        ...prev,
        brave_api_key: accepted.has("brave") ? "" : prev.brave_api_key,
        firecrawl_api_key: accepted.has("firecrawl") ? "" : prev.firecrawl_api_key,
        firecrawl_base_url: accepted.has("firecrawl") ? "" : prev.firecrawl_base_url,
        exa_api_key: accepted.has("exa") ? "" : prev.exa_api_key,
        tavily_api_key: accepted.has("tavily") ? "" : prev.tavily_api_key,
      }));
      setRemoveProviders((current) => current.filter((provider) => !accepted.has(provider)));
      setMessage({
        kind:
          next.operation_status === "rejected"
            ? "error"
            : next.operation_status === "partial"
              ? "warning"
              : "ok",
        text: next.banner?.message ?? `Search provider save: ${next.operation_status}.`,
      });
    } catch (err) {
      const text = err instanceof ApiError ? err.message : "Failed to save search providers.";
      setMessage({ kind: "error", text });
    } finally {
      setSaving(false);
    }
  };

  const toggleProvider = async (provider: SearchProviderId) => {
    if (!status?.search_providers?.[provider] || saving || testingTools) return;
    setSaving(true);
    setMessage(undefined);
    try {
      const current = status.search_providers[provider];
      const next = await api.toggleSearchProvider({
        expected_revision: status.provider_config_revision ?? 0,
        provider,
        enabled: current.enabled === false,
      });
      const nextStatus = {
        ...status,
        provider_config_revision: next.revision,
        search_providers: next.providers,
        external_search_service: next.external_search_service,
      };
      commitProviderStatus(nextStatus);
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof ApiError ? err.message : "Failed to update provider." });
    } finally {
      setSaving(false);
    }
  };

  const toggleRemove = (provider: SearchProviderId, field: keyof ProviderUpdate) => {
    setRemoveProviders((current) => {
      if (current.includes(provider)) return current.filter((item) => item !== provider);
      setForm((previous) => ({ ...previous, [field]: "" }));
      return [...current, provider];
    });
  };

  const saveSearchUsage = async () => {
    if (!searchUsage || searchUsageSaving) return;
    setSearchUsageSaving(true);
    setSearchUsageMessage(undefined);
    try {
      setSearchUsage(await api.updateSearchUsage(searchUsage));
      setSearchUsageMessage("Search usage policy saved.");
    } catch (err) {
      setSearchUsageMessage(err instanceof ApiError ? err.message : "Failed to save search usage policy.");
    } finally {
      setSearchUsageSaving(false);
    }
  };

  const saveDeepCheckpoint = async () => {
    if (!deepCheckpoint || deepCheckpointSaving) return;
    setDeepCheckpointSaving(true);
    setDeepCheckpointMessage(undefined);
    try {
      const next = await api.updateDeepCheckpointSettings(deepCheckpoint);
      setDeepCheckpoint(next);
      setDeepCheckpointMessage("Deep checkpoint preference saved for new runs.");
    } catch (err) {
      setDeepCheckpointMessage(
        err instanceof ApiError ? err.message : "Failed to save Deep checkpoint preference.",
      );
    } finally {
      setDeepCheckpointSaving(false);
    }
  };

  const confirmCachePurge = async () => {
    setCacheClearing(true);
    setCacheMessage(undefined);
    try {
      const result = await api.purgeCache({ scope: "all" });
      setCacheMessage({ kind: "ok", text: `Cleared ${result.purged_count} cache entries.` });
      setCacheDialogOpen(false);
    } catch (err) {
      const text = err instanceof ApiError ? err.message : "Failed to clear retrieval cache.";
      setCacheMessage({ kind: "error", text });
    } finally {
      setCacheClearing(false);
    }
  };

  return (
    <div ref={ref} className="mx-auto w-full max-w-[760px] px-6 py-8">
      <div data-anim className="mb-5 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-text">Settings</h1>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      </div>

      <div className="space-y-4">
        <form onSubmit={saveSearchProviders} aria-label="Search providers settings">
        <Card data-anim>
          <CardHeader>
            <CardTitle className="text-sm">Search providers</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Field
              label="Brave Search API key"
              type="password"
              autoComplete="off"
              placeholder={status?.brave_ready ? "•••••••• (leave blank to keep)" : "BSA..."}
              value={form.brave_api_key ?? ""}
              onChange={update("brave_api_key")}
              disabled={IS_DEMO || saving || testingTools || ["env", "secure"].includes(status?.search_providers?.brave.credential_source ?? "")}
            />
            <SearchProviderControls
              provider="brave"
              state={status?.search_providers?.brave}
              busy={IS_DEMO || saving || testingTools}
              removePending={removeProviders.includes("brave")}
              onToggle={() => void toggleProvider("brave")}
              onRemove={() => toggleRemove("brave", "brave_api_key")}
            />
            <Field
              label="Firecrawl API key"
              type="password"
              autoComplete="off"
              placeholder={status?.firecrawl_ready ? "•••••••• (leave blank to keep)" : "fc-..."}
              value={form.firecrawl_api_key ?? ""}
              onChange={update("firecrawl_api_key")}
              disabled={IS_DEMO || saving || testingTools || ["env", "secure"].includes(status?.search_providers?.firecrawl.credential_source ?? "")}
            />
            <Field
              label="Firecrawl base URL"
              placeholder="https://api.firecrawl.dev"
              value={form.firecrawl_base_url ?? ""}
              onChange={update("firecrawl_base_url")}
              disabled={IS_DEMO || saving || testingTools}
            />
            <SearchProviderControls
              provider="firecrawl"
              state={status?.search_providers?.firecrawl}
              busy={IS_DEMO || saving || testingTools}
              removePending={removeProviders.includes("firecrawl")}
              onToggle={() => void toggleProvider("firecrawl")}
              onRemove={() => toggleRemove("firecrawl", "firecrawl_api_key")}
            />
            <Field
              label="Tavily API key"
              hint="Search and Extract credits are billed by Tavily; connection tests may consume credits."
              type="password"
              autoComplete="off"
              placeholder={status?.tavily_ready ? "•••••••• (leave blank to keep)" : "tvly-..."}
              value={form.tavily_api_key ?? ""}
              onChange={update("tavily_api_key")}
              disabled={IS_DEMO || saving || testingTools || ["env", "secure"].includes(status?.search_providers?.tavily.credential_source ?? "")}
            />
            <SearchProviderControls
              provider="tavily"
              state={status?.search_providers?.tavily}
              busy={IS_DEMO || saving || testingTools}
              removePending={removeProviders.includes("tavily")}
              onToggle={() => void toggleProvider("tavily")}
              onRemove={() => toggleRemove("tavily", "tavily_api_key")}
            />
            <Field
              label="Exa API key"
              hint="Search and Contents credits are billed by Exa; connection tests may consume credits."
              type="password"
              autoComplete="off"
              placeholder={status?.exa_ready ? "•••••••• (leave blank to keep)" : "exa-..."}
              value={form.exa_api_key ?? ""}
              onChange={update("exa_api_key")}
              disabled={IS_DEMO || saving || testingTools || ["env", "secure"].includes(status?.search_providers?.exa.credential_source ?? "")}
            />
            <SearchProviderControls
              provider="exa"
              state={status?.search_providers?.exa}
              busy={IS_DEMO || saving || testingTools}
              removePending={removeProviders.includes("exa")}
              onToggle={() => void toggleProvider("exa")}
              onRemove={() => toggleRemove("exa", "exa_api_key")}
            />
            <div className="flex flex-wrap gap-2" aria-label="Test individual search provider">
              {(["brave", "firecrawl", "tavily", "exa"] as SearchProviderId[]).map((provider) => (
                <Button
                  key={provider}
                  variant="outline"
                  size="sm"
                  type="button"
                  onClick={() => void testTools(provider)}
                  disabled={IS_DEMO || testingTools || saving}
                >
                  <Plug className="h-4 w-4" /> Test {provider}
                </Button>
              ))}
            </div>
            {toolsTest && (
              <p className="text-xs text-text-muted" role="status">
                {"error" in toolsTest
                  ? toolsTest.error
                  : `${toolsTest.ok ? "✓" : "✗"} ${toolsTest.provider ?? "search"} ${toolsTest.status}`}
              </p>
            )}
            <p className="text-xs text-text-faint">
              Save and Test make real External search requests and may consume provider credits. Test is transient and never changes saved state.
            </p>
            {status?.external_search_service?.banner && (
              <p className="text-xs text-text-muted" role="status">
                {status.external_search_service.banner.message}
              </p>
            )}
            <div className="flex items-center gap-3">
              <Button type="submit" disabled={IS_DEMO || saving || testingTools}>
                <Save className="h-4 w-4" /> {saving ? "Saving…" : "Save search providers"}
              </Button>
              <span className="text-xs text-text-faint">
                Accepted config-owned credentials persist locally and apply to new runs immediately.
              </span>
            </div>
            {message && (
              <p
                className={`text-sm ${
                  message.kind === "ok"
                    ? "text-success"
                    : message.kind === "warning"
                      ? "text-warning"
                      : "text-danger"
                }`}
                role="status"
              >
                {message.text}
              </p>
            )}
          </CardContent>
        </Card>
        </form>

        <Card data-anim>
          <CardHeader>
            <CardTitle className="text-sm">Search usage</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {searchUsage ? (
              <SearchUsageEditor
                value={searchUsage}
                disabled={IS_DEMO || searchUsageSaving}
                capabilities={status?.search_providers}
                onChange={(policy) => policy && setSearchUsage(policy)}
              />
            ) : (
              <p className="text-sm text-text-muted">Loading search usage policy…</p>
            )}
            <div className="flex flex-wrap items-center gap-3">
              <Button type="button" size="sm" onClick={() => void saveSearchUsage()} disabled={IS_DEMO || !searchUsage || searchUsageSaving}>
                <Save className="h-4 w-4" /> {searchUsageSaving ? "Saving…" : "Save search usage"}
              </Button>
              {searchUsageMessage && <span role="status" className="text-xs text-text-muted">{searchUsageMessage}</span>}
            </div>
          </CardContent>
        </Card>

        <ProviderRegistry />

        <PostRunNotesEditor />

        <KnowledgeGraphSettingsEditor />

        <ContextBudgetEditor />

        <Card data-anim>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <CirclePause className="h-4 w-4" /> Deep checkpoints
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {deepCheckpoint ? (
              <label className="flex items-start gap-3">
                <input
                  type="checkbox"
                  className="mt-1 h-4 w-4 accent-[color:var(--accent)]"
                  checked={deepCheckpoint.auto_checkpoint_enabled}
                  onChange={(event) =>
                    setDeepCheckpoint({
                      auto_checkpoint_enabled: event.target.checked,
                    })
                  }
                  disabled={IS_DEMO || deepCheckpointSaving}
                />
                <span className="space-y-1">
                  <span className="block text-sm font-medium text-text">
                    Automatically save at safe Deep Research boundaries
                  </span>
                  <span className="block text-xs text-text-faint">
                    When off, only an explicit Pause creates a checkpoint. Unexpected interruption recovery is not available.
                  </span>
                </span>
              </label>
            ) : (
              <p className="text-sm text-text-muted">Loading Deep checkpoint preference…</p>
            )}
            <div className="flex flex-wrap items-center gap-3">
              <Button
                type="button"
                size="sm"
                onClick={() => void saveDeepCheckpoint()}
                disabled={IS_DEMO || !deepCheckpoint || deepCheckpointSaving}
              >
                <Save className="h-4 w-4" />
                {deepCheckpointSaving ? "Saving…" : "Save Deep checkpoint setting"}
              </Button>
              {deepCheckpointMessage ? (
                <span role="status" className="text-xs text-text-muted">
                  {deepCheckpointMessage}
                </span>
              ) : null}
            </div>
          </CardContent>
        </Card>

        <RetrievalSettingsEditor health={health} />

        <Card data-anim>
          <CardHeader>
            <CardTitle className="text-sm">System</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex flex-wrap items-center gap-3 text-sm text-text">
              <span className="flex items-center gap-1.5">
                <Server className="h-4 w-4 text-text-muted" aria-hidden />
                Job backend:
              </span>
              <span className="font-medium">
                {health?.job_backend === "rq"
                  ? "Queue (RQ)"
                  : health?.job_backend === "thread"
                    ? "In-process (thread)"
                    : "—"}
              </span>
              {health?.job_backend === "rq" && (
                <StatusPill
                  ok={!!health?.redis_reachable}
                  label={health?.redis_reachable ? "Redis connected" : "Redis unreachable"}
                />
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm text-text">
              <span className="flex items-center gap-1.5">
                <Database className="h-4 w-4 text-text-muted" aria-hidden />
                Database at rest:
              </span>
              <StatusPill
                ok={!!health?.db_encrypted}
                label={health?.db_encrypted ? "Encrypted (SQLCipher)" : "Plaintext"}
              />
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm text-text">
              <span className="flex items-center gap-1.5">
                <Boxes className="h-4 w-4 text-text-muted" aria-hidden />
                Vector search:
              </span>
              <span className="font-medium">
                {health?.vector_backend === "chroma"
                  ? "ANN (Chroma)"
                  : health?.vector_backend === "numpy"
                    ? "Vectorized (NumPy)"
                    : health?.vector_backend === "bruteforce"
                      ? "Brute force (SQLite)"
                      : "—"}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm text-text">
              <span className="flex items-center gap-1.5">
                <Lock className="h-4 w-4 text-text-muted" aria-hidden />
                API auth:
              </span>
              <StatusPill
                ok={health?.auth_mode === "token"}
                label={health?.auth_mode === "token" ? "Token required" : "Off (localhost)"}
              />
            </div>
            <div className="mt-3 border-t border-border pt-3">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="space-y-1">
                  <p className="text-sm font-medium text-text">Data & privacy</p>
                  <p className="max-w-xl text-xs text-text-faint">
                    Clear temporary retrieval cache without deleting sessions, runs, memories, Library items, or BYOK
                    settings.
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  type="button"
                  onClick={() => setCacheDialogOpen(true)}
                  disabled={IS_DEMO || cacheClearing}
                >
                  <Trash2 className="h-4 w-4" /> Clear retrieval cache
                </Button>
              </div>
              {cacheMessage && (
                <p
                  className={`mt-2 text-xs ${cacheMessage.kind === "ok" ? "text-success" : "text-danger"}`}
                  role="status"
                >
                  {cacheMessage.text}
                </p>
              )}
            </div>
            <p className="text-xs text-text-faint">
              Read-only. The job backend is set at deploy time — change it via
              DEEPRESEARCH_JOB_BACKEND in docker/.env and restart the API container.
              Database encryption is enabled by setting DEEPRESEARCH_DB_KEY. API auth is set via
              DEEPRESEARCH_AUTH_MODE.
            </p>
          </CardContent>
        </Card>

      </div>
      <Dialog
        open={cacheDialogOpen}
        onOpenChange={setCacheDialogOpen}
        title="Clear retrieval cache"
        description="This removes temporary search and fetch cache entries only."
      >
        <div className="space-y-4">
          <p className="text-sm text-text-muted">
            This does not delete sessions, runs, memories, Library items, or BYOK settings.
          </p>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={() => setCacheDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              type="button"
              variant="danger"
              size="sm"
              onClick={() => void confirmCachePurge()}
              disabled={IS_DEMO || cacheClearing}
            >
              <Trash2 className="h-4 w-4" /> {cacheClearing ? "Clearing…" : "Clear cache"}
            </Button>
          </div>
        </div>
      </Dialog>
    </div>
  );
}

function SearchProviderControls({
  provider,
  state,
  busy,
  removePending,
  onToggle,
  onRemove,
}: {
  provider: SearchProviderId;
  state?: SearchProviderCapability;
  busy: boolean;
  removePending: boolean;
  onToggle: () => void;
  onRemove: () => void;
}) {
  const lampClass = {
    green: "bg-success",
    yellow: "bg-warning",
    red: "bg-danger",
    gray: "bg-text-faint",
  }[state?.lamp ?? "gray"];
  const readOnly = state?.credential_source === "env" || state?.credential_source === "secure";
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
      <span className={`h-2 w-2 rounded-full ${lampClass}`} aria-hidden />
      <span className="capitalize">{provider}</span>
      <span>{state?.probe_state?.replaceAll("_", " ") ?? "not configured"}</span>
      <span>· {state?.credential_source ?? "none"}</span>
      <Button type="button" size="sm" variant="outline" aria-label={`${state?.enabled === false ? "Enable" : "Disable"} ${provider}`} disabled={busy || !state} onClick={onToggle}>
        <Power className="h-3.5 w-3.5" /> {state?.enabled === false ? "Enable" : "Disable"}
      </Button>
      <Button type="button" size="sm" variant="ghost" aria-label={`${removePending ? "Undo remove" : "Remove"} ${provider} credential`} disabled={busy || readOnly || !state?.configured} onClick={onRemove}>
        <Trash2 className="h-3.5 w-3.5" /> {removePending ? "Undo remove" : "Remove"}
      </Button>
      {readOnly && <span>Credential is managed by a read-only {state?.credential_source} source.</span>}
    </div>
  );
}
