import { useEffect, useState } from "react";
import { Save } from "lucide-react";
import type { Mode } from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { IS_DEMO } from "@/lib/runtime";

const MODES: { value: Mode; label: string }[] = [
  { value: "fast", label: "Fast Search" },
  { value: "multiple", label: "Multiple Search" },
  { value: "deep", label: "Deep Research" },
  { value: "council", label: "Models Council" },
];

export function PostRunNotesEditor() {
  const [enabled, setEnabled] = useState<Mode[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    api
      .getPostRunNotes()
      .then((config) => setEnabled(config.enabled_modes ?? []))
      .catch(() => setEnabled([]))
      .finally(() => setLoading(false));
  }, []);

  const toggle = (mode: Mode, next: boolean) => {
    setEnabled((current) => {
      if (next) return current.includes(mode) ? current : [...current, mode];
      return current.filter((item) => item !== mode);
    });
  };

  const save = async () => {
    setSaving(true);
    setMessage("");
    try {
      const config = await api.updatePostRunNotes({ enabled_modes: enabled });
      setEnabled(config.enabled_modes ?? []);
      setMessage("Saved post-run note settings.");
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : "Could not save post-run notes.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card data-anim>
      <CardHeader>
        <CardTitle className="text-sm">Post-run notes</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-text-muted">
          Disabled by default. Enable a mode only when it should write an automatic Library Note; Deep/Council
          reports and citation snapshots are not affected.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {MODES.map((mode) => {
            const checked = enabled.includes(mode.value);
            return (
              <div
                key={mode.value}
                className={cn(
                  "flex items-center justify-between rounded-md border p-3 transition-colors",
                  checked
                    ? "border-accent bg-accent-soft ring-2 ring-accent/25"
                    : "border-border hover:border-border-strong",
                )}
              >
                <span className={cn("text-sm font-medium", checked ? "text-accent" : "text-text")}>{mode.label}</span>
                <Switch
                  aria-label={`${mode.label} post-run notes`}
                  checked={checked}
                  disabled={IS_DEMO || loading || saving}
                  onCheckedChange={(next) => toggle(mode.value, next)}
                />
              </div>
            );
          })}
        </div>
        <div className="flex items-center gap-3">
          <Button size="sm" type="button" onClick={() => void save()} disabled={IS_DEMO || loading || saving}>
            <Save className="h-4 w-4" /> {saving ? "Saving..." : "Save notes"}
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
