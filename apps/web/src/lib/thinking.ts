import type { ThinkingEffort } from "./types";

/**
 * Effort <-> token-budget mapping, kept in sync with the backend
 * (deepresearch_core/baseline/llm_call_params.py). The UI speaks the coarse
 * Off/Low/Medium/High dial; the registry persists a token budget.
 */
export const THINKING_EFFORTS: ThinkingEffort[] = ["off", "low", "medium", "high"];

const LOW_BUDGET = 1024;
const MEDIUM_BUDGET = 4096;
const HIGH_BUDGET = 8192;

/** Map a persisted thinking token budget to its coarse effort label. */
export function budgetToEffort(budget: number | null | undefined): ThinkingEffort {
  if (!budget || budget <= 0) return "off";
  if (budget <= LOW_BUDGET) return "low";
  if (budget <= MEDIUM_BUDGET) return "medium";
  return "high";
}

/** Map a coarse effort label to a thinking token budget (null when off). */
export function effortToBudget(effort: ThinkingEffort): number | null {
  switch (effort) {
    case "low":
      return LOW_BUDGET;
    case "medium":
      return MEDIUM_BUDGET;
    case "high":
      return HIGH_BUDGET;
    default:
      return null;
  }
}

/** Human label for an effort value (for select options / hints). */
export function effortLabel(effort: ThinkingEffort): string {
  return { off: "Off", low: "Low", medium: "Medium", high: "High" }[effort];
}
