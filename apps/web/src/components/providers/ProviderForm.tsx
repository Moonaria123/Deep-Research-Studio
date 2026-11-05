import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import type { LLMProviderProfile } from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/components/ui/toast";
import { IS_DEMO } from "@/lib/runtime";

const PROVIDER_TYPES = ["openai", "ollama", "anthropic", "litellm"] as const;

interface ProviderFormProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** When set, the form edits an existing provider; otherwise it creates one. */
  provider?: LLMProviderProfile;
  /** Called with the fresh registry status after a successful save. */
  onSaved: (registry: import("@/lib/types").LLMRegistryStatus) => void;
}

interface Fields {
  id: string;
  label: string;
  type: string;
  base_url: string;
  api_key: string;
  enabled: boolean;
}

function looksLikeUrl(value: string): boolean {
  return /^https?:\/\/.+/i.test(value.trim());
}

/** Create/edit a provider at the provider level (models are managed separately). */
export function ProviderForm({ open, onOpenChange, provider, onSaved }: ProviderFormProps) {
  const isEdit = Boolean(provider);
  const [fields, setFields] = useState<Fields>({
    id: "",
    label: "",
    type: "openai",
    base_url: "",
    api_key: "",
    enabled: true,
  });
  const [errors, setErrors] = useState<Partial<Record<keyof Fields, string>>>({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open) return;
    setErrors({});
    if (provider) {
      setFields({
        id: provider.id,
        label: provider.label,
        type: provider.type,
        base_url: provider.base_url,
        api_key: "",
        enabled: provider.enabled,
      });
    } else {
      setFields({ id: "", label: "", type: "openai", base_url: "", api_key: "", enabled: true });
    }
  }, [open, provider]);

  const setField = <K extends keyof Fields>(key: K, value: Fields[K]) =>
    setFields((prev) => ({ ...prev, [key]: value }));

  const validate = (): boolean => {
    const next: Partial<Record<keyof Fields, string>> = {};
    if (!isEdit && !fields.id.trim()) next.id = "Provider ID is required.";
    if (!fields.base_url.trim()) next.base_url = "Base URL is required.";
    else if (!looksLikeUrl(fields.base_url)) next.base_url = "Base URL must start with http(s)://";
    if (!isEdit && fields.type !== "ollama" && !fields.api_key.trim()) {
      next.api_key = "API key is required for a new provider.";
    }
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const submit = async () => {
    if (!validate()) return;
    setSaving(true);
    try {
      let registry: import("@/lib/types").LLMRegistryStatus;
      if (isEdit && provider) {
        const patch: Partial<import("@/lib/types").LLMProviderUpsert> = {
          label: fields.label.trim(),
          type: fields.type,
          base_url: fields.base_url.trim(),
          enabled: fields.enabled,
        };
        if (fields.api_key.trim()) patch.api_key = fields.api_key.trim();
        registry = await api.patchLlmProvider(provider.id, patch);
        toast.success(`Updated provider "${provider.id}".`);
      } else {
        registry = await api.upsertLlmProvider({
          id: fields.id.trim(),
          label: fields.label.trim(),
          type: fields.type,
          base_url: fields.base_url.trim(),
          api_key: fields.api_key.trim() || undefined,
          enabled: fields.enabled,
          models: [],
        });
        toast.success(`Added provider "${fields.id.trim()}".`);
      }
      onSaved(registry);
      onOpenChange(false);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Save failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={isEdit ? `Edit provider · ${provider?.id}` : "Add provider"}
      description="Configure an OpenAI-compatible (or Ollama) endpoint. Manage its models separately."
    >
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1 text-xs text-text-muted">
            <span>Provider ID</span>
            <Input
              value={fields.id}
              onChange={(e) => setField("id", e.target.value)}
              placeholder="openai"
              aria-label="Provider ID"
              disabled={IS_DEMO || isEdit}
            />
            {errors.id && <span className="block text-danger">{errors.id}</span>}
          </label>
          <label className="space-y-1 text-xs text-text-muted">
            <span>Label</span>
            <Input
              value={fields.label}
              onChange={(e) => setField("label", e.target.value)}
              placeholder="OpenAI"
              aria-label="Label"
              disabled={IS_DEMO}
            />
          </label>
          <label className="space-y-1 text-xs text-text-muted">
            <span>Type</span>
            <select
              aria-label="Type"
              disabled={IS_DEMO}
              value={fields.type}
              onChange={(e) => setField("type", e.target.value)}
              className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
            >
              {PROVIDER_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs text-text-muted">
            <span>Base URL</span>
            <Input
              value={fields.base_url}
              onChange={(e) => setField("base_url", e.target.value)}
              placeholder="https://api.openai.com/v1"
              aria-label="Base URL"
              disabled={IS_DEMO}
            />
            {errors.base_url && <span className="block text-danger">{errors.base_url}</span>}
          </label>
          <label className="col-span-2 space-y-1 text-xs text-text-muted">
            <span>API key</span>
            <Input
              type="password"
              autoComplete="new-password"
              value={fields.api_key}
              onChange={(e) => setField("api_key", e.target.value)}
              placeholder={isEdit ? "leave blank to keep existing" : "sk-…"}
              aria-label="API key"
              disabled={IS_DEMO}
            />
            {IS_DEMO && <span className="block text-warning">Preview only — API keys are disabled.</span>}
            {isEdit && provider?.has_key && (
              <span className="block text-text-muted">A key is set. Enter a new key only to replace it.</span>
            )}
            {errors.api_key && <span className="block text-danger">{errors.api_key}</span>}
          </label>
        </div>

        <label className="flex items-center gap-2 text-xs text-text-muted">
          <Switch checked={fields.enabled} onCheckedChange={(v) => setField("enabled", v)} aria-label="Enabled" disabled={IS_DEMO} />
          Enabled
        </label>

        <div className="flex justify-end gap-2 pt-1">
          <Button variant="ghost" size="sm" type="button" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button size="sm" type="submit" disabled={IS_DEMO || saving}>
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {isEdit ? "Save changes" : "Add provider"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
