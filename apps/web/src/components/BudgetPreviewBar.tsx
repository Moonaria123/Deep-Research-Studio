import { Gauge, Search, Layers, FileText, Clock, DollarSign } from "lucide-react";
import type { Mode, RunCostEstimate } from "@/lib/types";
import { estimateBudget, formatDuration, type BudgetOverrides } from "@/lib/modeBudgetPresets";

/**
 * A compact, pre-submit estimate of the resource budget a run will use: searches,
 * top-k passages, an output-token ceiling, and an approximate time bound. Updates
 * live as the mode, search/top-k overrides, or council size change.
 */
export function BudgetPreviewBar({
  mode,
  overrides,
  costEstimate,
  costLoading,
}: {
  mode: Mode;
  overrides?: BudgetOverrides;
  costEstimate?: RunCostEstimate | null;
  costLoading?: boolean;
}) {
  const estimate = estimateBudget(mode, overrides);
  const totalCost = formatUsdRange(costEstimate?.total_usd_low, costEstimate?.total_usd_high);
  const searchCost = formatSearchCost(costEstimate);
  const llmCost = formatUsdRange(costEstimate?.llm_usd_low, costEstimate?.llm_usd_high);
  return (
    <div
      aria-label="Budget preview"
      className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-md border border-border bg-bg-subtle/60 px-3 py-1.5 text-xs text-text-muted"
      title={costEstimate?.disclaimers?.join(" ") || undefined}
    >
      <span className="flex items-center gap-1.5 font-medium text-text-body">
        <Gauge className="h-3.5 w-3.5" aria-hidden="true" />
        Estimated budget
      </span>
      <span className="flex items-center gap-1.5">
        <Search className="h-3.5 w-3.5" aria-hidden="true" />
        Searches <span className="font-medium text-text-body">{estimate.searches}</span>
      </span>
      {(costEstimate?.recovery_searches_max ?? (mode === "council" ? 0 : 1)) > 0 && (
        <span className="flex items-center gap-1.5">
          <Search className="h-3.5 w-3.5" aria-hidden="true" />
          Up to {costEstimate?.recovery_searches_max ?? 1} conditional focused recovery search
        </span>
      )}
      <span className="flex items-center gap-1.5">
        <Layers className="h-3.5 w-3.5" aria-hidden="true" />
        Top-K <span className="font-medium text-text-body">{estimate.topK}</span>
      </span>
      <span className="flex items-center gap-1.5">
        <FileText className="h-3.5 w-3.5" aria-hidden="true" />
        Output ≤ <span className="font-medium text-text-body">{estimate.outputTokens.toLocaleString()}</span> tokens
      </span>
      <span className="flex items-center gap-1.5">
        <Clock className="h-3.5 w-3.5" aria-hidden="true" />
        <span className="font-medium text-text-body">{formatDuration(estimate.timeoutS)}</span>
      </span>
      <span className="flex items-center gap-1.5">
        <DollarSign className="h-3.5 w-3.5" aria-hidden="true" />
        Cost{" "}
        <span className="font-medium text-text-body">
          {costLoading ? "estimating..." : totalCost ?? "unavailable"}
        </span>
      </span>
      {costEstimate && (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
          <span>Search {searchCost ?? "unavailable"}</span>
          <span>LLM {llmCost ?? "unavailable"}</span>
          <span>
            Tokens ~{costEstimate.llm_input_tokens_est.toLocaleString()} in / ~
            {costEstimate.llm_output_tokens_est.toLocaleString()} out
          </span>
        </span>
      )}
    </div>
  );
}

function formatUsdRange(low: number | null | undefined, high: number | null | undefined): string | null {
  if (low == null || high == null) return null;
  const left = formatUsd(low);
  const right = formatUsd(high);
  return left === right ? left : `${left}-${right}`;
}

function formatUsd(value: number): string {
  const digits = value > 0 && value < 0.01 ? 4 : 2;
  return `$${value.toFixed(digits)}`;
}

function formatSearchCost(costEstimate?: RunCostEstimate | null): string | null {
  if (!costEstimate) return null;
  const usd = formatUsdRange(costEstimate.search_usd_low, costEstimate.search_usd_high);
  if (usd) return usd;
  if (costEstimate.search_credits_low == null || costEstimate.search_credits_high == null) return null;
  const low = costEstimate.search_credits_low;
  const high = costEstimate.search_credits_high;
  return low === high ? `${low} credits` : `${low}-${high} credits`;
}
