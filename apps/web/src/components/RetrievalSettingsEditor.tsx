import { useEffect, useMemo, useState } from "react";
import { Save } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { RetrievalSettings } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { IS_DEMO } from "@/lib/runtime";

type HealthInfo = Awaited<ReturnType<typeof api.health>>;

export function RetrievalSettingsEditor({ health }: { health?: HealthInfo }) {
  const [settings, setSettings] = useState<RetrievalSettings>();
  const [enabled, setEnabled] = useState(true);
  const [kgEnabled, setKgEnabled] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    api
      .getRetrievalSettings()
      .then((next) => {
        setSettings(next);
        setEnabled(next.saved.chroma_persist_enabled);
        setKgEnabled(next.saved.kg_aware_retrieval_enabled ?? false);
      })
      .catch(() => undefined)
      .finally(() => setLoading(false));
  }, []);

  const chromaAvailable = health?.vector_backend === "chroma";
  const effective = settings?.effective;
  const disabledReason = useMemo(() => {
    if (!chromaAvailable) return "Chroma backend is not active; this setting is saved for the next Chroma runtime.";
    if (effective?.ephemeral_forced) return "Ephemeral mode is forced by the current runtime.";
    return "";
  }, [chromaAvailable, effective?.ephemeral_forced]);

  const save = async () => {
    setSaving(true);
    setMessage("");
    try {
      const next = await api.updateRetrievalSettings({
        chroma_persist_enabled: enabled,
        kg_aware_retrieval_enabled: kgEnabled,
      });
      setSettings(next);
      setEnabled(next.saved.chroma_persist_enabled);
      setKgEnabled(next.saved.kg_aware_retrieval_enabled ?? false);
      setMessage(next.restart_required ? "Saved. Restart the API for this change to take effect." : "Saved.");
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : "Could not save retrieval settings.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card data-anim>
      <CardHeader>
        <CardTitle className="text-sm">Retrieval infrastructure</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col gap-3 rounded-md border border-border p-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0 space-y-1">
            <p className="text-sm font-medium text-text">Persist Chroma vector index on disk</p>
            <p className="text-xs text-text-muted">
              Reuses the local vector cache after API restarts. SQLite embeddings remain the source of truth.
            </p>
            <p className="text-xs text-text-faint">
              Effective mode: {effective?.chroma_persist_mode ?? health?.chroma_persist_mode ?? "unknown"}
              {effective?.chroma_persist_path ? ` · ${effective.chroma_persist_path}` : ""}
            </p>
            {disabledReason && <p className="text-xs text-text-faint">{disabledReason}</p>}
          </div>
          <Switch
            aria-label="Persist Chroma vector index on disk"
            checked={enabled}
            disabled={IS_DEMO || loading || saving || !chromaAvailable || !!effective?.ephemeral_forced}
            onCheckedChange={setEnabled}
          />
        </div>
        <div className="flex flex-col gap-3 rounded-md border border-border p-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0 space-y-1">
            <p className="text-sm font-medium text-text">Use the knowledge graph for candidate discovery</p>
            <p className="text-xs text-text-muted">
              Adds bounded graph expansion to Multiple, Deep, and Council runs. Fast remains unchanged.
            </p>
            <p className="text-xs text-text-faint">
              Graph candidates still hydrate to local, Memory, or external evidence and never replace External search.
            </p>
          </div>
          <Switch
            aria-label="Use the knowledge graph for candidate discovery"
            checked={kgEnabled}
            disabled={IS_DEMO || loading || saving}
            onCheckedChange={setKgEnabled}
          />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button size="sm" type="button" onClick={() => void save()} disabled={IS_DEMO || loading || saving}>
            <Save className="h-4 w-4" /> {saving ? "Saving..." : "Save retrieval"}
          </Button>
          {message ? (
            <span className="text-xs text-text-muted" role="status">
              {message}
            </span>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}
