import { useEffect, useMemo, useState } from "react";
import { FlaskConical, Plus, Save } from "lucide-react";
import type { LLMProviderProfile, LLMRegistryStatus, LLMTestResult, MemoryFormatProbeResult, MemorySettings } from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { chatRefs, embedRefs } from "@/lib/registry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/components/ui/toast";
import { ProviderForm } from "@/components/providers/ProviderForm";
import { ProviderTable } from "@/components/providers/ProviderTable";
import { ModelsManageDialog } from "@/components/providers/ModelsManageDialog";
import { IS_DEMO } from "@/lib/runtime";

type TestState = LLMTestResult | { error: string };

const DEFAULT_MEMORY_SETTINGS: MemorySettings = {
  post_turn_memory_refresh_enabled: true,
  workspace_memory_refresh_strategy: "incremental",
  memory_thinking_enabled: false,
  format_probe: null,
};

function probeVariant(status?: MemoryFormatProbeResult["status"]) {
  if (status === "passed") return "success";
  if (status === "caution") return "warning";
  if (status === "failed") return "danger";
  return "default";
}

function probeBadgeVariant(probe?: MemoryFormatProbeResult | null) {
  if (probe?.stale) return "warning";
  return probeVariant(probe?.status);
}

function probeLabel(probe?: MemoryFormatProbeResult | null) {
  if (probe?.stale) return "Stale";
  if (probe?.status === "passed") return "Passed";
  if (probe?.status === "caution") return "Caution";
  if (probe?.status === "failed") return "Failed";
  return "Not tested";
}

/** Multi-provider LLM registry manager: table + create/edit dialog + model manager. */
export function ProviderRegistry() {
  const [registry, setRegistry] = useState<LLMRegistryStatus>();
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<LLMProviderProfile>();
  const [modelsFor, setModelsFor] = useState<LLMProviderProfile>();
  const [deleting, setDeleting] = useState<LLMProviderProfile>();
  const [tests, setTests] = useState<Record<string, TestState>>({});
  const [memorySettings, setMemorySettings] = useState<MemorySettings>(DEFAULT_MEMORY_SETTINGS);
  const [memorySettingsBusy, setMemorySettingsBusy] = useState(false);
  const [memoryProbe, setMemoryProbe] = useState<MemoryFormatProbeResult | null>(null);
  const [memoryProbeBusy, setMemoryProbeBusy] = useState(false);
  const [roles, setRoles] = useState<{ default: string; embed: string; memory: string; council: string[] }>({
    default: "",
    embed: "",
    memory: "",
    council: [],
  });

  const applyRegistry = (next: LLMRegistryStatus) => {
    setRegistry(next);
    setRoles({
      default: next.roles.default,
      embed: next.roles.embed,
      memory: next.roles.memory ?? "",
      council: [...next.roles.council],
    });
  };

  const load = () => api.llmRegistry().then(applyRegistry).catch(() => undefined);
  const loadMemorySettings = () =>
    api
      .getMemorySettings()
      .then((settings) => {
        setMemorySettings(settings);
        setMemoryProbe(settings.format_probe);
      })
      .catch(() => undefined);

  useEffect(() => {
    void load();
    void loadMemorySettings();
  }, []);

  const chatModelRefs = useMemo(() => chatRefs(registry), [registry]);
  const embedModelRefs = useMemo(() => embedRefs(registry), [registry]);
  const providers = registry?.providers ?? [];

  const openCreate = () => {
    setEditing(undefined);
    setFormOpen(true);
  };

  const openEdit = (provider: LLMProviderProfile) => {
    setEditing(provider);
    setFormOpen(true);
  };

  const toggleProvider = async (provider: LLMProviderProfile, enabled: boolean) => {
    try {
      applyRegistry(await api.patchLlmProvider(provider.id, { enabled }));
      toast.success(`${enabled ? "Enabled" : "Disabled"} "${provider.id}".`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update provider.");
    }
  };

  const testProvider = async (id: string) => {
    setTests((prev) => ({ ...prev, [id]: { error: "Testing…" } }));
    try {
      const result = await api.testLlm({ provider_id: id });
      setTests((prev) => ({ ...prev, [id]: result }));
      if (result.ok) toast.success(`"${id}" reachable.`);
      else toast.error(`"${id}": ${result.status}`);
    } catch (err) {
      const text = err instanceof ApiError ? err.message : "test failed";
      setTests((prev) => ({ ...prev, [id]: { error: text } }));
      toast.error(text);
    }
  };

  const confirmDelete = async () => {
    if (!deleting) return;
    const id = deleting.id;
    try {
      applyRegistry(await api.removeLlmProvider(id));
      toast.success(`Removed provider "${id}".`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : `Could not remove "${id}".`);
    } finally {
      setDeleting(undefined);
    }
  };

  const saveRoles = async () => {
    try {
      applyRegistry(
        await api.setLlmRoles({
          default: roles.default,
          embed: roles.embed,
          memory: roles.memory,
          council: roles.council,
        }),
      );
      toast.success("Saved model roles.");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not save roles.");
    }
  };

  const testMemoryFormat = async () => {
    if (!roles.memory || memoryProbeBusy) return;
    setMemoryProbeBusy(true);
    try {
      applyRegistry(
        await api.setLlmRoles({
          default: roles.default,
          embed: roles.embed,
          memory: roles.memory,
          council: roles.council,
        }),
      );
      const result = await api.probeMemoryFormat();
      setMemoryProbe(result);
      if (result.status === "passed") toast.success("Memory format test passed.");
      else if (result.status === "caution") toast.info("Memory format test needs fallback.");
      else toast.error("Memory format test failed.");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not test memory format.");
    } finally {
      setMemoryProbeBusy(false);
    }
  };

  const toggleMemoryThinking = async (enabled: boolean) => {
    setMemorySettingsBusy(true);
    try {
      const next = await api.updateMemorySettings({ memory_thinking_enabled: enabled });
      setMemorySettings(next);
      setMemoryProbe(next.format_probe);
      toast.success(`Memory thinking ${enabled ? "enabled" : "disabled"}.`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update memory thinking.");
    } finally {
      setMemorySettingsBusy(false);
    }
  };

  const currentProbe = memoryProbe?.provider_ref === roles.memory ? memoryProbe : null;

  return (
    <Card data-anim>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="text-sm">LLM providers</CardTitle>
        <Button variant="outline" size="sm" type="button" onClick={openCreate}>
          <Plus className="h-4 w-4" /> Add provider
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {providers.length === 0 ? (
          <p className="text-sm text-text-faint">No LLM providers configured yet. Add one to get started.</p>
        ) : (
          <ProviderTable
            providers={providers}
            tests={tests}
            onToggle={toggleProvider}
            onTest={testProvider}
            onManageModels={setModelsFor}
            onEdit={openEdit}
            onDelete={setDeleting}
          />
        )}

        {providers.length > 0 && (
          <div className="space-y-3 rounded-md border border-border bg-bg-subtle p-3">
            <span className="text-sm font-medium text-text">Model roles</span>
            <p className="text-xs text-text-faint">
              Council members are chosen per question in the composer. These roles set the default chat model, embedder,
              and memory processing model.
            </p>
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="space-y-1 text-xs text-text-muted">
                <span>Default model</span>
                <select
                  aria-label="Default model"
                  value={roles.default}
                  disabled={IS_DEMO}
                  onChange={(e) => setRoles((prev) => ({ ...prev, default: e.target.value }))}
                  className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
                >
                  <option value="">(none)</option>
                  {chatModelRefs.map((r) => (
                    <option key={r.ref} value={r.ref}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1 text-xs text-text-muted">
                <span>Memory processing model</span>
                <select
                  aria-label="Memory processing model"
                  value={roles.memory}
                  disabled={IS_DEMO}
                  onChange={(e) => setRoles((prev) => ({ ...prev, memory: e.target.value }))}
                  className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
                >
                  <option value="">(none)</option>
                  {chatModelRefs.map((r) => (
                    <option key={r.ref} value={r.ref}>
                      {r.label}
                    </option>
                  ))}
                </select>
                <span className="block text-[11px] leading-snug text-text-faint">
                  Prefer a non-reasoning small model with stable JSON and instruction following.
                </span>
                <div className="flex items-center justify-between gap-3 rounded-sm bg-surface px-2 py-1.5 text-[11px] text-text-muted">
                  <span>Use thinking for memory refresh</span>
                  <span className="flex items-center gap-2">
                    <span className="text-text-body">{memorySettings.memory_thinking_enabled ? "On" : "Off"}</span>
                    <Switch
                      checked={memorySettings.memory_thinking_enabled}
                      disabled={IS_DEMO || memorySettingsBusy}
                      aria-label="Memory thinking"
                      onCheckedChange={(checked) => void toggleMemoryThinking(checked)}
                    />
                  </span>
                </div>
                <span className="block text-[11px] leading-snug text-text-faint">
                  Format tests keep thinking off to verify baseline JSON compatibility.
                </span>
                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => void testMemoryFormat()}
                    disabled={IS_DEMO || !roles.memory || memoryProbeBusy}
                  >
                    <FlaskConical className="h-4 w-4" />
                    {memoryProbeBusy ? "Testing" : "Test memory format"}
                  </Button>
                  <Badge variant={probeBadgeVariant(currentProbe)}>{probeLabel(currentProbe)}</Badge>
                </div>
                {currentProbe?.stale ? (
                  <span className="block text-[11px] leading-snug text-warning">
                    Memory model settings changed. Test memory format again.
                  </span>
                ) : null}
                {currentProbe?.status === "failed" ? (
                  <span className="block text-[11px] leading-snug text-danger">
                    Choose a memory model with stable JSON output.
                  </span>
                ) : null}
              </label>
              <label className="space-y-1 text-xs text-text-muted">
                <span>Embedding model</span>
                <select
                  aria-label="Embedding model"
                  value={roles.embed}
                  disabled={IS_DEMO}
                  onChange={(e) => setRoles((prev) => ({ ...prev, embed: e.target.value }))}
                  className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
                >
                  <option value="">(default / local)</option>
                  {embedModelRefs.map((r) => (
                    <option key={r.ref} value={r.ref}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <Button size="sm" type="button" onClick={saveRoles} disabled={IS_DEMO}>
              <Save className="h-4 w-4" /> Save roles
            </Button>
          </div>
        )}
      </CardContent>

      <ProviderForm open={formOpen} onOpenChange={setFormOpen} provider={editing} onSaved={applyRegistry} />
      <ModelsManageDialog
        open={Boolean(modelsFor)}
        onOpenChange={(open) => !open && setModelsFor(undefined)}
        provider={modelsFor}
        onChanged={load}
      />
      <Dialog
        open={Boolean(deleting)}
        onOpenChange={(open) => !open && setDeleting(undefined)}
        title="Delete provider"
        description={deleting ? `Remove "${deleting.id}" and clear any roles bound to it? This cannot be undone.` : ""}
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
    </Card>
  );
}
