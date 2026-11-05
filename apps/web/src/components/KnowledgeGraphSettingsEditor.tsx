import { useEffect, useState } from "react";
import { Save } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { IS_DEMO } from "@/lib/runtime";

export function KnowledgeGraphSettingsEditor() {
  const [enabled, setEnabled] = useState(true);
  const [wikiAuto, setWikiAuto] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    api
      .getKnowledgeGraphSettings()
      .then((config) => {
        setEnabled(config.post_run_rebuild_enabled);
        setWikiAuto(config.wiki_auto_refresh_after_kg_rebuild);
      })
      .catch(() => {
        setEnabled(true);
        setWikiAuto(false);
      })
      .finally(() => setLoading(false));
  }, []);

  const save = async () => {
    setSaving(true);
    setMessage("");
    try {
      const config = await api.updateKnowledgeGraphSettings({
        post_run_rebuild_enabled: enabled,
        wiki_auto_refresh_after_kg_rebuild: wikiAuto,
      });
      setEnabled(config.post_run_rebuild_enabled);
      setWikiAuto(config.wiki_auto_refresh_after_kg_rebuild);
      setMessage("Saved knowledge graph settings.");
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : "Could not save knowledge graph settings.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card data-anim>
      <CardHeader>
        <CardTitle className="text-sm">Knowledge graph</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between gap-4 rounded-md border border-border p-3">
          <div className="min-w-0">
            <p className="text-sm font-medium text-text">Refresh after completed research</p>
            <p className="mt-1 text-xs text-text-muted">
              Keeps the local research graph current after completed or partial runs with usable output.
            </p>
          </div>
          <Switch
            aria-label="Refresh knowledge graph after completed research"
            checked={enabled}
            disabled={IS_DEMO || loading || saving}
            onCheckedChange={setEnabled}
          />
        </div>
        <div className="flex items-center justify-between gap-4 rounded-md border border-border p-3">
          <div className="min-w-0">
            <p className="text-sm font-medium text-text">Refresh Wiki Map after graph rebuild</p>
            <p className="mt-1 text-xs text-text-muted">
              Starts a Wiki Map refresh only after a knowledge graph rebuild successfully commits a new active version.
            </p>
          </div>
          <Switch
            aria-label="Refresh Wiki Map after knowledge graph rebuild"
            checked={wikiAuto}
            disabled={IS_DEMO || loading || saving}
            onCheckedChange={setWikiAuto}
          />
        </div>
        <div className="flex items-center gap-3">
          <Button size="sm" type="button" onClick={() => void save()} disabled={IS_DEMO || loading || saving}>
            <Save className="h-4 w-4" /> {saving ? "Saving..." : "Save graph"}
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
