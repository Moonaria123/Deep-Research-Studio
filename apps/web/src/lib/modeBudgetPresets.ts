import type { CouncilStrategy, Mode } from "./types";
import generatedPresets from "./modeBudgetPresets.generated.json";

export interface ModeBudgetPreset {
  max_searches: number;
  max_output_tokens: number;
  timeout_s: number;
  top_k: number;
}

export const MODE_BUDGET_PRESETS = generatedPresets as Record<Mode, ModeBudgetPreset>;

export interface BudgetOverrides {
  maxSearches?: number;
  topK?: number;
  councilSize?: number;
  /** Council aggregation strategy, used to preview output + time differences. */
  councilStrategy?: CouncilStrategy;
}

// Fastest council path: no chair synthesis, just select the best member draft.
const PARALLEL_VOTE_FACTOR = 0.85;
// Conservative multiplier for the extra critique round (DR-ITER-044).
const DEBATE_ONCE_FACTOR = 1.3;

function councilStrategyFactor(strategy: CouncilStrategy | undefined): number {
  if (strategy === "parallel_vote") return PARALLEL_VOTE_FACTOR;
  if (strategy === "debate_once") return DEBATE_ONCE_FACTOR;
  return 1;
}

export interface BudgetEstimate {
  searches: number;
  outputTokens: number;
  timeoutS: number;
  topK: number;
}

/** Compute the effective budget for a mode, applying any Composer overrides. */
export function estimateBudget(mode: Mode, overrides: BudgetOverrides = {}): BudgetEstimate {
  const preset = MODE_BUDGET_PRESETS[mode];
  const searches =
    overrides.maxSearches != null && Number.isFinite(overrides.maxSearches)
      ? overrides.maxSearches
      : preset.max_searches;
  const topK =
    overrides.topK != null && Number.isFinite(overrides.topK) ? overrides.topK : preset.top_k;
  // Council runs synthesize across members, so the output budget scales with size —
  // a conservative member-count factor rather than an exact upstream figure.
  const sizeFactor = mode === "council" ? Math.max(1, overrides.councilSize ?? 3) : 1;
  const strategyFactor =
    mode === "council" ? councilStrategyFactor(overrides.councilStrategy) : 1;
  return {
    searches,
    outputTokens: Math.round(preset.max_output_tokens * sizeFactor * strategyFactor),
    timeoutS: Math.round(preset.timeout_s * strategyFactor),
    topK,
  };
}

/** Humanize a seconds budget into a compact duration label (e.g. "~30s", "~3m"). */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `~${seconds}s`;
  return `~${Math.round(seconds / 60)}m`;
}
