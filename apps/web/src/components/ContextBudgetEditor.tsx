import { useEffect, useMemo, useState } from "react";
import { Save, Sparkles, RotateCcw, Check } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { ContextBudgetRecommendation, ContextBudgetState } from "@/lib/types";
import {
  CONTEXT_BLOCKS,
  CONTEXT_PRESETS,
  PRESET_SHARES,
  type ContextPresetId,
} from "@/lib/contextBudget";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { IS_DEMO } from "@/lib/runtime";

interface Draft {
  preset: string;
  recent_share: number;
  semantic_share: number;
  memory_share: number;
  max_chars: number;
  turns: number;
}

function isNamedPreset(id: string): id is ContextPresetId {
  return CONTEXT_PRESETS.some((p) => p.id === id);
}

function draftFromShares(preset: string, shares: [number, number, number], max_chars: number, turns: number): Draft {
  return {
    preset,
    recent_share: shares[0],
    semantic_share: shares[1],
    memory_share: shares[2],
    max_chars,
    turns,
  };
}

/** Settings card for the three-block context-injection budget (DR-ITER-057).
 *
 * Presets + custom shares + a stacked-bar preview + a manual heuristic
 * "smart recommend" flow. Saves a global default to config.json; per-run
 * overrides live in the Composer. Self-contained: loads + persists its own state.
 */
export function ContextBudgetEditor() {
  const [state, setState] = useState<ContextBudgetState>();
  const [draft, setDraft] = useState<Draft>();
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string }>();
  const [recommendation, setRecommendation] = useState<ContextBudgetRecommendation>();
  const [recommending, setRecommending] = useState(false);

  const initFrom = (next: ContextBudgetState) => {
    const eff = next.effective;
    setDraft({
      preset: eff.preset,
      recent_share: eff.recent_share,
      semantic_share: eff.semantic_share,
      memory_share: eff.memory_share,
      max_chars: eff.max_chars,
      turns: eff.turns,
    });
  };

  useEffect(() => {
    api
      .getContextBudget()
      .then((next) => {
        setState(next);
        initFrom(next);
      })
      .catch(() => undefined);
  }, []);

  const shareSum = draft ? draft.recent_share + draft.semantic_share + draft.memory_share : 0;
  const sharesValid = Math.abs(shareSum - 100) <= 1 && (draft?.recent_share ?? 0) > 0;
  const isCustom = draft ? !isNamedPreset(draft.preset) : false;

  const selectPreset = (id: ContextPresetId) =>
    setDraft((d) => (d ? draftFromShares(id, PRESET_SHARES[id], d.max_chars, d.turns) : d));

  const setShare = (key: "recent" | "semantic" | "memory", value: number) =>
    setDraft((d) =>
      d ? { ...d, preset: "custom", [`${key}_share`]: Math.max(0, Math.min(100, value)) } : d,
    );

  const recommend = async () => {
    setRecommending(true);
    setMessage(undefined);
    try {
      const rec = await api.recommendContextBudget({});
      setRecommendation(rec);
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof ApiError ? err.message : "Recommend failed" });
    } finally {
      setRecommending(false);
    }
  };

  const adoptRecommendation = () => {
    if (!recommendation || !draft) return;
    const r = recommendation.recommendation;
    setDraft({
      preset: r.preset,
      recent_share: r.recent_share,
      semantic_share: r.semantic_share,
      memory_share: r.memory_share,
      max_chars: r.max_chars,
      turns: r.turns,
    });
    setRecommendation(undefined);
  };

  const restoreDefault = () => {
    if (state) initFrom({ ...state, effective: state.default });
  };

  const save = async () => {
    if (!draft) return;
    setSaving(true);
    setMessage(undefined);
    try {
      const payload = isNamedPreset(draft.preset)
        ? { preset: draft.preset, max_chars: draft.max_chars, turns: draft.turns }
        : {
            preset: "custom",
            recent_share: draft.recent_share,
            semantic_share: draft.semantic_share,
            memory_share: draft.memory_share,
            max_chars: draft.max_chars,
            turns: draft.turns,
          };
      const next = await api.updateContextBudget(payload);
      setState(next);
      initFrom(next);
      setMessage({ kind: "ok", text: "Saved. New research runs will use this global budget." });
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof ApiError ? err.message : "Save failed" });
    } finally {
      setSaving(false);
    }
  };

  const barShares = useMemo(
    () =>
      draft
        ? { recent: draft.recent_share, semantic: draft.semantic_share, memory: draft.memory_share }
        : { recent: 0, semantic: 0, memory: 0 },
    [draft],
  );

  if (!draft) {
    return (
      <Card data-anim>
        <CardHeader>
          <CardTitle className="text-sm">Context continuity budget</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-text-faint">Loading…</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-anim>
      <CardHeader>
        <CardTitle className="text-sm">Context continuity budget</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-text-muted">
          Controls three context quotas injected into the prompt. Fill order is fixed: recent
          conversation → related earlier turns → long-term memory. When a block has no content, its
          quota rolls forward to earlier blocks.
        </p>

        {/* Preset cards */}
        <div className="grid grid-cols-2 gap-2">
          {CONTEXT_PRESETS.map((p) => {
            const active = draft.preset === p.id;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => selectPreset(p.id)}
                disabled={IS_DEMO}
                aria-pressed={active}
                data-selected={active ? "true" : undefined}
                className={cn(
                  "rounded-md border p-2 text-left text-xs shadow-card transition-colors focus-ring",
                  active
                    ? "border-accent bg-accent-soft ring-2 ring-accent/25"
                    : "border-border bg-surface text-text-muted hover:border-border-strong",
                )}
              >
                <span className={cn("block font-semibold", active ? "text-accent" : "text-text")}>
                  {p.label}
                </span>
                <span className={cn("mt-0.5 block", active ? "text-text-body" : "text-text-faint")}>
                  {p.description}
                </span>
              </button>
            );
          })}
        </div>
        {isCustom && (
          <p className="text-xs font-medium text-primary">Custom allocation</p>
        )}

        {/* Stacked bar */}
        <div>
          <div
            aria-label="Budget allocation preview"
            className="flex h-4 w-full overflow-hidden rounded-full border border-border"
          >
            {CONTEXT_BLOCKS.map((b) => (
              <div
                key={b.key}
                style={{ width: `${barShares[b.key]}%`, backgroundColor: b.color }}
                title={`${b.label} ${barShares[b.key]}%`}
              />
            ))}
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-muted">
            {CONTEXT_BLOCKS.map((b) => (
              <span key={b.key} className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: b.color }} />
                {b.label} <span className="font-medium text-text-body">{barShares[b.key]}%</span>
              </span>
            ))}
          </div>
        </div>

        {/* Custom sliders */}
        <div className="space-y-2">
          {CONTEXT_BLOCKS.map((b) => (
            <label key={b.key} className="block space-y-1 text-xs text-text-muted">
              <span className="flex items-center justify-between">
                <span>
                  {b.label} <span className="text-text-faint">— {b.hint}</span>
                </span>
                <span className="font-medium text-text-body">{barShares[b.key]}%</span>
              </span>
              <input
                type="range"
                min={0}
                max={100}
                value={barShares[b.key]}
                aria-label={`${b.label} share`}
                onChange={(e) => setShare(b.key, Number(e.target.value))}
                disabled={IS_DEMO}
                className="w-full"
              />
            </label>
          ))}
          <p className={`text-xs ${sharesValid ? "text-text-faint" : "text-danger"}`}>
            Total {shareSum}% (must equal 100%)
          </p>
        </div>

        {/* Advanced: total + turns */}
        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1 text-xs text-text-muted">
            <span>Total character budget (≈ {Math.round(draft.max_chars / 4)} tokens)</span>
            <Input
              type="number"
              min={1000}
              step={500}
              value={draft.max_chars}
              aria-label="Total character budget"
              disabled={IS_DEMO}
              onChange={(e) =>
                setDraft((d) => (d ? { ...d, max_chars: Number(e.target.value) || d.max_chars } : d))
              }
            />
          </label>
          <label className="space-y-1 text-xs text-text-muted">
            <span>Recent turn count</span>
            <Input
              type="number"
              min={1}
              max={20}
              value={draft.turns}
              aria-label="Recent turn count"
              disabled={IS_DEMO}
              onChange={(e) =>
                setDraft((d) => (d ? { ...d, turns: Number(e.target.value) || d.turns } : d))
              }
            />
          </label>
        </div>

        {/* Smart recommend */}
        <div className="rounded-md border border-border bg-bg-subtle p-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-text">Smart recommend</span>
            <Button type="button" variant="outline" size="sm" onClick={recommend} disabled={IS_DEMO || recommending}>
              <Sparkles className="h-3.5 w-3.5" /> {recommending ? "Analyzing…" : "Smart recommend"}
            </Button>
          </div>
          {recommendation && (
            <div className="mt-2 space-y-2 text-xs text-text-muted" role="status">
              <p className="text-text">
                Recommended preset:{" "}
                <span className="font-medium">
                  {CONTEXT_PRESETS.find((p) => p.id === recommendation.recommendation.preset)?.label ??
                    "Custom"}
                </span>
                （{recommendation.recommendation.recent_share}/
                {recommendation.recommendation.semantic_share}/
                {recommendation.recommendation.memory_share}）
              </p>
              <ul className="list-disc space-y-0.5 pl-4">
                {recommendation.reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
              <Button type="button" variant="outline" size="sm" onClick={adoptRecommendation} disabled={IS_DEMO}>
                <Check className="h-3.5 w-3.5" /> Adopt
              </Button>
            </div>
          )}
        </div>

        {message && (
          <p className={`text-xs ${message.kind === "ok" ? "text-success" : "text-danger"}`} role="status">
            {message.text}
          </p>
        )}

        <div className="flex items-center gap-3">
          <Button type="button" onClick={save} disabled={IS_DEMO || saving || (isCustom && !sharesValid)}>
            <Save className="h-4 w-4" /> {saving ? "Saving…" : "Save budget"}
          </Button>
          <Button type="button" variant="ghost" size="sm" onClick={restoreDefault} disabled={IS_DEMO}>
            <RotateCcw className="h-4 w-4" /> Restore default
          </Button>
          {state?.source === "env" && (
            <span className="text-xs text-text-faint">Active source: deployment env fallback</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
