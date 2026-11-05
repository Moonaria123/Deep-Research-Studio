import { describe, expect, it } from "vitest";
import generatedPresets from "./modeBudgetPresets.generated.json";
import { estimateBudget, formatDuration, MODE_BUDGET_PRESETS } from "./modeBudgetPresets";

describe("estimateBudget", () => {
  it("uses the generated backend preset export", () => {
    expect(MODE_BUDGET_PRESETS).toEqual(generatedPresets);
  });

  it("returns the preset values when no overrides are given", () => {
    const e = estimateBudget("deep");
    expect(e.searches).toBe(MODE_BUDGET_PRESETS.deep.max_searches);
    expect(e.topK).toBe(MODE_BUDGET_PRESETS.deep.top_k);
    expect(e.outputTokens).toBe(MODE_BUDGET_PRESETS.deep.max_output_tokens);
    expect(e.timeoutS).toBe(MODE_BUDGET_PRESETS.deep.timeout_s);
  });

  it("applies finite search and top-k overrides", () => {
    const e = estimateBudget("fast", { maxSearches: 9, topK: 12 });
    expect(e.searches).toBe(9);
    expect(e.topK).toBe(12);
  });

  it("ignores non-finite overrides", () => {
    const e = estimateBudget("fast", { maxSearches: NaN, topK: undefined });
    expect(e.searches).toBe(MODE_BUDGET_PRESETS.fast.max_searches);
    expect(e.topK).toBe(MODE_BUDGET_PRESETS.fast.top_k);
  });

  it("scales council output tokens by member count", () => {
    expect(estimateBudget("council", { councilSize: 4 }).outputTokens).toBe(
      MODE_BUDGET_PRESETS.council.max_output_tokens * 4,
    );
    // size factor never drops below 1.
    expect(estimateBudget("council", { councilSize: 0 }).outputTokens).toBe(
      MODE_BUDGET_PRESETS.council.max_output_tokens,
    );
  });

  it("does not scale output tokens for non-council modes", () => {
    expect(estimateBudget("deep", { councilSize: 5 }).outputTokens).toBe(
      MODE_BUDGET_PRESETS.deep.max_output_tokens,
    );
  });

  it("estimates distinct output and timeout for each council strategy", () => {
    const base = estimateBudget("council", { councilSize: 3 });
    const parallel = estimateBudget("council", { councilSize: 3, councilStrategy: "parallel_vote" });
    const judge = estimateBudget("council", { councilSize: 3, councilStrategy: "judge_synthesis" });
    const debate = estimateBudget("council", { councilSize: 3, councilStrategy: "debate_once" });

    expect(judge).toEqual(base);
    expect(parallel.outputTokens).toBe(Math.round(base.outputTokens * 0.85));
    expect(parallel.timeoutS).toBe(Math.round(base.timeoutS * 0.85));
    expect(debate.outputTokens).toBe(Math.round(base.outputTokens * 1.3));
    expect(debate.timeoutS).toBe(Math.round(base.timeoutS * 1.3));
  });

  it("ignores the council strategy for non-council modes", () => {
    expect(estimateBudget("deep", { councilStrategy: "debate_once" }).outputTokens).toBe(
      MODE_BUDGET_PRESETS.deep.max_output_tokens,
    );
  });
});

describe("formatDuration", () => {
  it("renders sub-minute as seconds", () => {
    expect(formatDuration(30)).toBe("~30s");
  });

  it("rounds to whole minutes above 60s", () => {
    expect(formatDuration(120)).toBe("~2m");
    expect(formatDuration(180)).toBe("~3m");
  });
});
