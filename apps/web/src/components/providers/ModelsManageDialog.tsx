import { Fragment, useEffect, useMemo, useState } from "react";
import { CloudDownload, Loader2, Pencil, Plug, Plus, Trash2 } from "lucide-react";
import type {
  LLMProviderProfile,
  LLMTestResult,
  ModelKind,
  ModelProfile,
  ModelProfileInput,
  ThinkingEffort,
} from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { budgetToEffort, effortToBudget, THINKING_EFFORTS, effortLabel } from "@/lib/thinking";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { toast } from "@/components/ui/toast";
import { IS_DEMO } from "@/lib/runtime";

interface ModelsManageDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  provider: LLMProviderProfile | undefined;
  /** Refresh the parent registry after model changes (counts/roles may shift). */
  onChanged: () => void;
}

function modelBadge(model: ModelProfile): string {
  if (model.kind === "embed") return "Embed";
  if (model.capabilities?.vision || model.capabilities?.ocr) return "Multimodal";
  return "Text";
}

/** Capability/category draft shared by the manual-add form and inline edit. */
interface ModelDraft {
  alias: string;
  kind: ModelKind;
  contextWindow: string;
  enabled: boolean;
  vision: boolean;
  ocr: boolean;
  webSearch: boolean;
  /** Default thinking depth; drives both ``thinking_budget`` and ``capabilities.thinking``. */
  thinkingDepth: ThinkingEffort;
  /** Default sampling temperature ("" = unset / provider default). */
  temperature: string;
}

/** Per-model chat-probe state (pending / transport error / secret-free result). */
type ModelTestState = { pending: true } | { error: string } | LLMTestResult;

const EMPTY_DRAFT: ModelDraft = {
  alias: "",
  kind: "chat",
  contextWindow: "",
  enabled: true,
  vision: false,
  ocr: false,
  webSearch: false,
  thinkingDepth: "off",
  temperature: "",
};

function toDraft(model: ModelProfile): ModelDraft {
  return {
    alias: model.alias ?? "",
    kind: model.kind,
    contextWindow: model.context_window ? String(model.context_window) : "",
    enabled: model.enabled,
    vision: model.capabilities?.vision ?? false,
    ocr: model.capabilities?.ocr ?? false,
    webSearch: model.capabilities?.web_search ?? false,
    thinkingDepth: budgetToEffort(model.thinking_budget),
    temperature: model.temperature != null ? String(model.temperature) : "",
  };
}

/** Translate a draft into the API model-profile payload (capabilities + context). */
function draftToBody(draft: ModelDraft): Omit<ModelProfileInput, "id"> {
  const thinkingOn = draft.thinkingDepth !== "off";
  return {
    alias: draft.alias.trim(),
    kind: draft.kind,
    context_window: draft.contextWindow.trim() ? Number(draft.contextWindow) : 0,
    enabled: draft.enabled,
    thinking_budget: effortToBudget(draft.thinkingDepth),
    temperature: draft.temperature.trim() ? Number(draft.temperature) : null,
    capabilities: {
      vision: draft.vision,
      ocr: draft.ocr,
      web_search: draft.webSearch,
      thinking: thinkingOn,
    },
  };
}

/**
 * Per-provider model manager (SRB "Model management" dialog):
 * 1. Fetch upstream catalog, then pick one via filter + dropdown ("Add selected model").
 * 2. Add a model manually with full parameters.
 * 3. Manage configured models inline (edit capabilities / chat-test / delete).
 */
export function ModelsManageDialog({ open, onOpenChange, provider, onChanged }: ModelsManageDialogProps) {
  const [models, setModels] = useState<ModelProfile[]>([]);
  const [loading, setLoading] = useState(false);

  // Upstream fetch + single-select picker.
  const [upstream, setUpstream] = useState<string[]>([]);
  const [upstreamFilter, setUpstreamFilter] = useState("");
  const [selectedUpstreamId, setSelectedUpstreamId] = useState("");
  const [fetching, setFetching] = useState(false);
  const [fetched, setFetched] = useState(false);

  // Manual add form.
  const [manualId, setManualId] = useState("");
  const [manualDraft, setManualDraft] = useState<ModelDraft>(EMPTY_DRAFT);
  const [adding, setAdding] = useState(false);

  const [modelTests, setModelTests] = useState<Record<string, ModelTestState>>({});

  // Inline edit (configured models table).
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState<ModelDraft | null>(null);
  const [saving, setSaving] = useState(false);

  const providerId = provider?.id ?? "";

  useEffect(() => {
    if (!open) {
      setUpstream([]);
      setUpstreamFilter("");
      setSelectedUpstreamId("");
      setFetched(false);
      setManualId("");
      setManualDraft(EMPTY_DRAFT);
      setModelTests({});
      setEditingId(null);
      setEditDraft(null);
      return;
    }
    if (!providerId) return;
    setLoading(true);
    api
      .listProviderModels(providerId)
      .then((res) => setModels(res.models))
      .catch((err) => toast.error(err instanceof ApiError ? err.message : "Could not load models."))
      .finally(() => setLoading(false));
  }, [open, providerId]);

  const existingIds = useMemo(() => new Set(models.map((m) => m.id)), [models]);

  // Filter applies to the whole catalog; the dropdown only offers not-yet-added ids.
  const upstreamFiltered = useMemo(() => {
    const q = upstreamFilter.trim().toLowerCase();
    return q ? upstream.filter((id) => id.toLowerCase().includes(q)) : upstream;
  }, [upstream, upstreamFilter]);
  const selectable = useMemo(
    () => upstreamFiltered.filter((id) => !existingIds.has(id)),
    [upstreamFiltered, existingIds],
  );
  const alreadyAddedCount = upstreamFiltered.length - selectable.length;

  // Keep the dropdown selection valid as the filter/catalog/configured set changes.
  useEffect(() => {
    setSelectedUpstreamId((current) =>
      current && selectable.includes(current) ? current : selectable[0] ?? "",
    );
  }, [selectable]);

  const fetchUpstream = async () => {
    if (!providerId) return;
    setFetching(true);
    try {
      const res = await api.fetchModels({ provider_id: providerId });
      if (!res.ok) {
        toast.error(res.error_message || `Upstream fetch failed (${res.status}).`);
        return;
      }
      setUpstream(res.models);
      setFetched(true);
      if (res.models.length === 0) toast.info("Provider returned no models.");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not reach the provider.");
    } finally {
      setFetching(false);
    }
  };

  const addSelected = async () => {
    const id = selectedUpstreamId;
    if (!providerId || !id || existingIds.has(id)) return;
    setAdding(true);
    try {
      const res = await api.createProviderModels(providerId, { models: [{ id }] });
      setModels(res.models);
      onChanged();
      toast.success(`Added model "${id}".`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not add model.");
    } finally {
      setAdding(false);
    }
  };

  const addManual = async () => {
    const id = manualId.trim();
    if (!id || !providerId) return;
    if (existingIds.has(id)) {
      toast.info(`Model "${id}" already exists.`);
      return;
    }
    setAdding(true);
    try {
      const res = await api.createProviderModels(providerId, {
        models: [{ id, ...draftToBody(manualDraft) }],
      });
      setModels(res.models);
      setManualId("");
      setManualDraft(EMPTY_DRAFT);
      onChanged();
      toast.success(`Added model "${id}".`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not add model.");
    } finally {
      setAdding(false);
    }
  };

  const testModel = async (modelId: string) => {
    if (!providerId) return;
    setModelTests((prev) => ({ ...prev, [modelId]: { pending: true } }));
    try {
      const res = await api.testLlm({ provider_id: providerId, probe: "chat", model: modelId });
      setModelTests((prev) => ({ ...prev, [modelId]: res }));
      if (res.ok) toast.success(`"${modelId}" reachable.`);
      else toast.error(`"${modelId}": ${res.status}`);
    } catch (err) {
      const text = err instanceof ApiError ? err.message : "Chat test failed.";
      setModelTests((prev) => ({ ...prev, [modelId]: { error: text } }));
      toast.error(text);
    }
  };

  const startEdit = (model: ModelProfile) => {
    setEditingId(model.id);
    setEditDraft(toDraft(model));
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditDraft(null);
  };

  const saveEdit = async () => {
    if (!providerId || !editingId || !editDraft) return;
    setSaving(true);
    try {
      const res = await api.updateProviderModel(providerId, editingId, draftToBody(editDraft));
      setModels(res.models);
      onChanged();
      toast.success(`Updated model "${editingId}".`);
      cancelEdit();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update model.");
    } finally {
      setSaving(false);
    }
  };

  const remove = async (modelId: string) => {
    if (!providerId) return;
    try {
      await api.deleteProviderModel(providerId, modelId);
      setModels((prev) => prev.filter((m) => m.id !== modelId));
      onChanged();
      toast.success(`Removed model "${modelId}".`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not remove model.");
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Model management"
      description={provider?.label || provider?.id || ""}
      className="max-w-3xl"
    >
      <fieldset className="space-y-4" disabled={IS_DEMO} aria-label="Model management controls">
        {IS_DEMO ? (
          <p className="rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning">
            Preview only — model discovery, editing, testing, and deletion are disabled.
          </p>
        ) : null}
        {/* Card A — Fetch upstream catalog, pick one via filter + dropdown. */}
        <section className="rounded-md border border-border bg-bg-subtle p-3">
          <p className="flex items-center gap-2 text-sm font-medium text-text">
            <CloudDownload className="h-4 w-4" /> Fetch models from provider
          </p>
          <p className="mt-1 text-xs text-text-faint">
            Calls GET {"{base}"}/models with the provider base URL and API key (OpenAI Bearer / Anthropic
            x-api-key). Pick a model below to add it; nothing is fetched into your config until you add it.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              type="button"
              onClick={() => void fetchUpstream()}
              disabled={fetching}
            >
              {fetching ? <Loader2 className="h-4 w-4 animate-spin" /> : <CloudDownload className="h-4 w-4" />}
              Fetch available models
            </Button>
            {fetched && upstream.length > 0 && (
              <>
                <Input
                  value={upstreamFilter}
                  onChange={(e) => setUpstreamFilter(e.target.value)}
                  placeholder="Filter model ID…"
                  aria-label="Filter upstream models"
                  className="h-9 w-44 text-sm"
                />
                <select
                  value={selectedUpstreamId}
                  onChange={(e) => setSelectedUpstreamId(e.target.value)}
                  aria-label="Upstream models"
                  disabled={selectable.length === 0}
                  className="h-9 min-w-[12rem] flex-1 rounded-md border border-border bg-surface px-2 text-sm text-text disabled:opacity-50"
                >
                  {selectable.length === 0 ? (
                    <option value="">No selectable models</option>
                  ) : (
                    selectable.map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))
                  )}
                </select>
                <Button
                  size="sm"
                  type="button"
                  onClick={() => void addSelected()}
                  disabled={adding || !selectedUpstreamId}
                >
                  {adding ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                  Add selected model
                </Button>
              </>
            )}
          </div>
          {fetched && upstream.length === 0 && (
            <p className="mt-2 text-xs text-text-faint">
              Provider returned no models. Add a model id manually below.
            </p>
          )}
          {fetched && upstream.length > 0 && (
            <p className="mt-2 text-xs text-text-faint">
              {selectable.length} selectable
              {alreadyAddedCount > 0 ? ` · ${alreadyAddedCount} already configured` : ""}
            </p>
          )}
        </section>

        {/* Card B — Add a model manually with full parameters. */}
        <section className="rounded-md border border-border p-3">
          <p className="flex items-center gap-2 text-sm font-medium text-text">
            <Plus className="h-4 w-4" /> Add model manually
          </p>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <label className="space-y-1 text-xs text-text-muted">
              <span>modelId</span>
              <Input
                value={manualId}
                onChange={(e) => setManualId(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void addManual();
                  }
                }}
                placeholder="gpt-4o-mini"
                aria-label="Add model id"
              />
            </label>
            <label className="space-y-1 text-xs text-text-muted">
              <span>Alias (optional)</span>
              <Input
                value={manualDraft.alias}
                onChange={(e) => setManualDraft({ ...manualDraft, alias: e.target.value })}
                placeholder="Display name"
                aria-label="Manual alias"
              />
            </label>
            <label className="space-y-1 text-xs text-text-muted">
              <span>Kind</span>
              <select
                value={manualDraft.kind}
                onChange={(e) => setManualDraft({ ...manualDraft, kind: e.target.value as ModelKind })}
                aria-label="Manual kind"
                className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
              >
                <option value="chat">chat</option>
                <option value="embed">embed</option>
              </select>
            </label>
            <label className="space-y-1 text-xs text-text-muted">
              <span>Context window (tokens)</span>
              <Input
                type="number"
                min={0}
                value={manualDraft.contextWindow}
                onChange={(e) => setManualDraft({ ...manualDraft, contextWindow: e.target.value })}
                placeholder="128000"
                aria-label="Manual context window"
              />
            </label>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-text-muted">
            <label className="flex items-center gap-2">
              <Switch
                checked={manualDraft.enabled}
                onCheckedChange={(v) => setManualDraft({ ...manualDraft, enabled: v })}
                aria-label="Manual enabled"
              />
              Enabled
            </label>
            <label className="flex items-center gap-2">
              <Switch
                checked={manualDraft.vision}
                onCheckedChange={(v) => setManualDraft({ ...manualDraft, vision: v })}
                aria-label="Manual vision capability"
              />
              Vision
            </label>
            <label className="flex items-center gap-2">
              <Switch
                checked={manualDraft.ocr}
                onCheckedChange={(v) => setManualDraft({ ...manualDraft, ocr: v })}
                aria-label="Manual OCR capability"
              />
              OCR
            </label>
            <label className="flex items-center gap-2">
              <Switch
                checked={manualDraft.webSearch}
                onCheckedChange={(v) => setManualDraft({ ...manualDraft, webSearch: v })}
                aria-label="Manual web search capability"
              />
              Web search
            </label>
            <Button
              size="sm"
              type="button"
              className="ml-auto"
              onClick={() => void addManual()}
              disabled={adding || !manualId.trim()}
            >
              {adding ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />} Add
            </Button>
          </div>
          {manualDraft.kind === "chat" && (
            <div className="mt-3 grid grid-cols-2 gap-3">
              <label className="space-y-1 text-xs text-text-muted">
                <span>Temperature</span>
                <Input
                  type="number"
                  min={0}
                  max={2}
                  step={0.1}
                  value={manualDraft.temperature}
                  onChange={(e) => setManualDraft({ ...manualDraft, temperature: e.target.value })}
                  placeholder="provider default"
                  aria-label="Manual default temperature"
                />
              </label>
              <label className="space-y-1 text-xs text-text-muted">
                <span>Default thinking depth</span>
                <select
                  value={manualDraft.thinkingDepth}
                  onChange={(e) =>
                    setManualDraft({ ...manualDraft, thinkingDepth: e.target.value as ThinkingEffort })
                  }
                  aria-label="Manual thinking depth"
                  className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
                >
                  {THINKING_EFFORTS.map((eff) => (
                    <option key={eff} value={eff}>
                      {effortLabel(eff)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
        </section>

        {/* Card C — Configured models (DB-backed), inline edit + chat test. */}
        {loading ? (
          <p className="py-6 text-center text-sm text-text-faint">Loading models…</p>
        ) : models.length === 0 ? (
          <p className="py-6 text-center text-sm text-text-faint">No models yet. Add one above.</p>
        ) : (
          <div className="overflow-x-auto rounded-md border border-border">
            <Table>
              <THead>
                <TR>
                  <TH>Model</TH>
                  <TH>Kind</TH>
                  <TH className="text-center">Context</TH>
                  <TH>Status</TH>
                  <TH className="text-right">Actions</TH>
                </TR>
              </THead>
              <TBody>
                {models.map((m) => {
                  const isEditing = editingId === m.id && editDraft;
                  return (
                    <Fragment key={m.id}>
                      <TR>
                        <TD>
                          <div className="font-medium text-text">{m.alias || m.id}</div>
                          {m.alias && <div className="text-xs text-text-faint">{m.id}</div>}
                          {m.kind === "chat" && (m.temperature != null || (m.thinking_budget ?? 0) > 0) && (
                            <div className="text-xs text-text-faint">
                              {m.temperature != null ? `temp ${m.temperature}` : "temp default"}
                              {" · "}
                              {`think ${effortLabel(budgetToEffort(m.thinking_budget))}`}
                            </div>
                          )}
                          <ModelTestLine state={modelTests[m.id]} />
                        </TD>
                        <TD>
                          <Badge>{modelBadge(m)}</Badge>
                        </TD>
                        <TD className="text-center text-xs text-text-muted">
                          {m.context_window || "—"}
                        </TD>
                        <TD>
                          <span className={m.enabled ? "text-success" : "text-text-faint"}>
                            {m.enabled ? "Enabled" : "Disabled"}
                          </span>
                        </TD>
                        <TD>
                          <div className="flex justify-end gap-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              type="button"
                              aria-label={`Test chat ${m.id}`}
                              disabled={Boolean(modelTests[m.id] && "pending" in modelTests[m.id])}
                              onClick={() => void testModel(m.id)}
                            >
                              {modelTests[m.id] && "pending" in modelTests[m.id] ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Plug className="h-4 w-4" />
                              )}
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              type="button"
                              aria-label={`Edit model ${m.id}`}
                              onClick={() => (isEditing ? cancelEdit() : startEdit(m))}
                            >
                              <Pencil className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              type="button"
                              aria-label={`Remove model ${m.id}`}
                              onClick={() => void remove(m.id)}
                            >
                              <Trash2 className="h-4 w-4 text-danger" />
                            </Button>
                          </div>
                        </TD>
                      </TR>
                      {isEditing && (
                        <TR>
                          <TD colSpan={5} className="bg-accent-soft/20">
                            <ModelEditPanel
                              draft={editDraft}
                              onChange={setEditDraft}
                              onCancel={cancelEdit}
                              onSave={() => void saveEdit()}
                              saving={saving}
                              placeholder={m.id}
                            />
                          </TD>
                        </TR>
                      )}
                    </Fragment>
                  );
                })}
              </TBody>
            </Table>
          </div>
        )}
      </fieldset>
    </Dialog>
  );
}

/** Inline edit panel for a configured model (capabilities + category + context). */
function ModelEditPanel({
  draft,
  onChange,
  onCancel,
  onSave,
  saving,
  placeholder,
}: {
  draft: ModelDraft;
  onChange: (draft: ModelDraft) => void;
  onCancel: () => void;
  onSave: () => void;
  saving: boolean;
  placeholder: string;
}) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <label className="space-y-1 text-xs text-text-muted">
          <span>Alias</span>
          <Input
            value={draft.alias}
            onChange={(e) => onChange({ ...draft, alias: e.target.value })}
            placeholder={placeholder}
            aria-label="Alias"
          />
        </label>
        <label className="space-y-1 text-xs text-text-muted">
          <span>Kind</span>
          <select
            value={draft.kind}
            onChange={(e) => onChange({ ...draft, kind: e.target.value as ModelKind })}
            aria-label="Kind"
            className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
          >
            <option value="chat">chat</option>
            <option value="embed">embed</option>
          </select>
        </label>
        <label className="space-y-1 text-xs text-text-muted">
          <span>Context window</span>
          <Input
            type="number"
            min={0}
            value={draft.contextWindow}
            onChange={(e) => onChange({ ...draft, contextWindow: e.target.value })}
            placeholder="128000"
            aria-label="Context window"
          />
        </label>
        <label className="flex items-center gap-2 self-end text-xs text-text-muted">
          <Switch
            checked={draft.enabled}
            onCheckedChange={(v) => onChange({ ...draft, enabled: v })}
            aria-label="Model enabled"
          />
          Enabled
        </label>
      </div>
      <div className="flex flex-wrap gap-4 text-xs text-text-muted">
        <label className="flex items-center gap-2">
          <Switch
            checked={draft.vision}
            onCheckedChange={(v) => onChange({ ...draft, vision: v })}
            aria-label="Vision capability"
          />
          Vision
        </label>
        <label className="flex items-center gap-2">
          <Switch
            checked={draft.ocr}
            onCheckedChange={(v) => onChange({ ...draft, ocr: v })}
            aria-label="OCR capability"
          />
          OCR
        </label>
        <label className="flex items-center gap-2">
          <Switch
            checked={draft.webSearch}
            onCheckedChange={(v) => onChange({ ...draft, webSearch: v })}
            aria-label="Web search capability"
          />
          Web search
        </label>
      </div>
      {draft.kind === "chat" && (
        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1 text-xs text-text-muted">
            <span>Default temperature</span>
            <Input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={draft.temperature}
              onChange={(e) => onChange({ ...draft, temperature: e.target.value })}
              placeholder="provider default"
              aria-label="Default temperature"
            />
          </label>
          <label className="space-y-1 text-xs text-text-muted">
            <span>Default thinking depth</span>
            <select
              value={draft.thinkingDepth}
              onChange={(e) => onChange({ ...draft, thinkingDepth: e.target.value as ThinkingEffort })}
              aria-label="Thinking depth"
              className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
            >
              {THINKING_EFFORTS.map((eff) => (
                <option key={eff} value={eff}>
                  {effortLabel(eff)}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" type="button" onClick={onCancel}>
          Cancel
        </Button>
        <Button size="sm" type="button" onClick={onSave} disabled={saving}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Save
        </Button>
      </div>
    </div>
  );
}

/** Inline secret-free result of a per-model chat probe (under the model id). */
function ModelTestLine({ state }: { state: ModelTestState | undefined }) {
  if (!state) return null;
  if ("pending" in state) {
    return <div className="text-xs text-text-faint">Testing chat…</div>;
  }
  if ("error" in state) {
    return (
      <div className="text-xs text-danger" role="status">
        ✗ {state.error}
      </div>
    );
  }
  const latency = state.latency_ms ? ` · ${state.latency_ms}ms` : "";
  const detail = state.error_message?.trim() ? ` · ${state.error_message.trim()}` : "";
  return (
    <div className={`text-xs ${state.ok ? "text-success" : "text-danger"}`} role="status">
      {state.ok ? "✓" : "✗"} {state.status}
      {detail}
      {latency}
    </div>
  );
}
