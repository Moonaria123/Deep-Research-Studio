import type { ContextBudgetConfig } from "./types";

/** Named context-budget presets (DR-ITER-057), shared by Settings + Composer. */
export const CONTEXT_PRESETS = [
  {
    id: "balanced",
    label: "Balanced",
    description: "Balanced split (default): recent conversation first, with semantic recall and long-term memory.",
  },
  {
    id: "followup",
    label: "Follow-up focus",
    description: "Favors recent conversation: best for follow-ups tied to the current thread.",
  },
  {
    id: "long_thread",
    label: "Long thread",
    description: "Favors semantic recall: connects earlier relevant turns in long sessions.",
  },
  {
    id: "memory_rich",
    label: "Memory rich",
    description: "Favors long-term memory: use when workspace background and preferences are rich.",
  },
] as const;

export type ContextPresetId = (typeof CONTEXT_PRESETS)[number]["id"];

/** Canonical (recent, semantic, memory) shares per preset, for live UI preview.
 * The authoritative caps are recomputed by the backend on save. */
export const PRESET_SHARES: Record<ContextPresetId, [number, number, number]> = {
  balanced: [58, 25, 17],
  followup: [70, 20, 10],
  long_thread: [50, 35, 15],
  memory_rich: [45, 15, 40],
};

/** Human label for a preset id (falls back to "Custom" for unknown/custom). */
export function presetLabel(preset: string): string {
  return CONTEXT_PRESETS.find((p) => p.id === preset)?.label ?? "Custom";
}

/** The three continuity blocks, in fixed fill order, for the stacked bar + sliders. */
export const CONTEXT_BLOCKS: {
  key: "recent" | "semantic" | "memory";
  label: string;
  hint: string;
  color: string;
}[] = [
  {
    key: "recent",
    label: "Recent conversation",
    hint: "Linear transcript of the latest turns in this session",
    color: "var(--color-primary, #2563eb)",
  },
  {
    key: "semantic",
    label: "Related earlier turns",
    hint: "Earlier turns in this session via semantic recall (requires embedding model)",
    color: "#8b5cf6",
  },
  {
    key: "memory",
    label: "Long-term memory",
    hint: "Workspace/user preferences and background (not usable as citation sources)",
    color: "#0d9488",
  },
];

/** Share value (%) for a block from a config. */
export function blockShare(config: ContextBudgetConfig, key: "recent" | "semantic" | "memory"): number {
  return config[`${key}_share` as keyof ContextBudgetConfig] as number;
}

/** Char cap for a block from a config. */
export function blockChars(config: ContextBudgetConfig, key: "recent" | "semantic" | "memory"): number {
  return config[`${key}_chars` as keyof ContextBudgetConfig] as number;
}
