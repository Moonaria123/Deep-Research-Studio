import { describe, expect, it } from "vitest";
import { runResultToMarkdown, slugifyQuery } from "./exportMarkdown";
import type { EvidenceSource, RunResult } from "./types";

const sources: EvidenceSource[] = [
  { chunk_id: 1, document_id: "d1", url: "https://e.com/a", title: "Source A", text: "", score: 0.9 },
  { chunk_id: 2, document_id: "d2", url: "", title: "Untitled", text: "", score: 0.5 },
];

const baseResult: RunResult = {
  run_id: "r1",
  mode: "fast",
  status: "succeeded",
  answer: { text: "Water boils at 100C [c1].", citations: [], claim_map: {}, coverage: 1 },
  sources,
};

describe("runResultToMarkdown", () => {
  it("includes the query heading, answer body and numbered sources", () => {
    const md = runResultToMarkdown("Boiling point of water", baseResult, sources);
    expect(md).toContain("# Boiling point of water");
    expect(md).toContain("Water boils at 100C [c1].");
    expect(md).toContain("## Sources");
    expect(md).toContain("1. [Source A](https://e.com/a)");
    // A source without a URL renders as plain text.
    expect(md).toContain("2. Untitled");
  });

  it("marks low-confidence exports and lists every item to verify before the answer", () => {
    const partial: RunResult = {
      ...baseResult,
      status: "completed_degraded",
      verification: { status: "completed" },
      confidence_warning: { reasons: ["evidence_insufficient"], acknowledged: false },
      unverified_items: [
        { question: "Was the feature generally available?", reason: "evidence_insufficient" },
        { question: "Did all sources agree?", reason: "source_conflict" },
      ],
    };

    const md = runResultToMarkdown("Feature status", partial, sources);

    expect(md).toContain("> **Low-confidence warning**");
    expect(md).toContain("## Items to verify");
    expect(md).toContain("Was the feature generally available? — Evidence was insufficient.");
    expect(md).toContain("Did all sources agree? — Sources remain in conflict.");
    expect(md.indexOf("## Items to verify")).toBeLessThan(md.indexOf("Water boils at 100C"));
  });

  it("discloses when verification is still running at export time", () => {
    const md = runResultToMarkdown(
      "Feature status",
      { ...baseResult, verification: { status: "running" } },
      sources,
    );

    expect(md).toContain("> **Verification pending**");
  });

  it("demotes deep-research sections into a Research process appendix", () => {
    const deep: RunResult = {
      ...baseResult,
      mode: "deep",
      answer: { text: "Final cohesive report [c1].", citations: [], claim_map: {}, coverage: 1 },
      sections: [{ title: "Background", answer: { text: "Some background.", citations: [], claim_map: {}, coverage: 1 } }],
    };
    const md = runResultToMarkdown("Topic", deep, []);
    // The final report leads; section drafts follow under a Research process appendix.
    expect(md).toContain("Final cohesive report [c1].");
    expect(md).toContain("## Research process");
    expect(md).toContain("### 1. Background");
    expect(md).toContain("Some background.");
    expect(md.indexOf("Final cohesive report")).toBeLessThan(md.indexOf("## Research process"));
  });

  it("appends a separate Limitations block and a confidence rating", () => {
    const withLimits: RunResult = {
      ...baseResult,
      answer: {
        text: "Water boils at 100C [c1].",
        citations: [],
        claim_map: {},
        coverage: 0.9,
        limitations: "Pressure assumptions vary [c1].",
        confidence_label: "established",
      },
    };
    const md = runResultToMarkdown("Boiling point", withLimits, []);
    expect(md).toContain("> Confidence rating: Established");
    expect(md).toContain("## Limitations");
    expect(md).toContain("Pressure assumptions vary [c1].");
    // The limitation is not inside the body line.
    expect(md.indexOf("## Limitations")).toBeGreaterThan(md.indexOf("Water boils at 100C"));
  });

  it("appends a Search lanes section for multiple mode", () => {
    const multiple: RunResult = {
      ...baseResult,
      mode: "multiple",
      answer: { text: "Merged answer [c1].", citations: [], claim_map: {}, coverage: 1 },
      lanes: [
        { query: "sub query one", sources: [sources[0]] },
        { query: "sub query two", sources: [] },
      ],
    };
    const md = runResultToMarkdown("Topic", multiple, sources);
    expect(md).toContain("Merged answer [c1].");
    expect(md).toContain("## Search lanes");
    expect(md).toContain("### 1. sub query one");
    expect(md).toContain("- [Source A](https://e.com/a)");
    expect(md).toContain("### 2. sub query two");
    expect(md).toContain("- (no sources)");
    expect(md.indexOf("Merged answer")).toBeLessThan(md.indexOf("## Search lanes"));
  });

  it("does not export lane evidence rejected from the final snapshot", () => {
    const multiple: RunResult = {
      ...baseResult,
      mode: "multiple",
      sources: [sources[0]],
      lanes: [{ query: "focused query", sources }],
    };
    const md = runResultToMarkdown("Topic", multiple, [sources[0]]);

    expect(md).toContain("Source A");
    expect(md).not.toContain("Untitled");
  });

  it("renders council members", () => {
    const council: RunResult = {
      ...baseResult,
      mode: "council",
      confidence: 0.8,
      members: [{ model: "model-x", answer: { text: "Consensus view.", citations: [], claim_map: {}, coverage: 1 } }],
    };
    const md = runResultToMarkdown("Topic", council, []);
    expect(md).toContain("Confidence: 80%");
    expect(md).toContain("## model-x");
    expect(md).toContain("Consensus view.");
  });

  it("uses answer conflicts as the canonical Council export source without duplication", () => {
    const conflict = {
      id: "conflict-new",
      claim_id: "claim-new",
      claim: "Canonical Council conflict",
      type: "peer_unresolved" as const,
      claim_scope: "same scope",
      evidence_ids: ["c1", "c2"],
      resolution: "contested" as const,
      reason: "The evidence remains tied.",
    };
    const council: RunResult = {
      ...baseResult,
      mode: "council",
      answer: { ...baseResult.answer, conflicts: [conflict], limitations: "Council limitation." },
      conflicts: [{ ...conflict, id: "conflict-legacy", claim: "Legacy duplicate" }],
      members: [{ model: "model-x", answer: baseResult.answer }],
      temporal_coverage: {
        intent: "latest",
        checked_at: 10,
        status: "unknown",
        basis: "none",
        evidence_ids: [],
        reason_codes: ["temporal_coverage_unknown"],
        statement: "Current coverage is not fully bounded.",
        note_required: true,
      },
    };

    const md = runResultToMarkdown("Topic", council, sources);
    const order = [
      "> 时效范围声明：",
      "Water boils at 100C",
      "## 证据冲突附注",
      "## 时效性附注",
      "## Limitations",
      "## model-x",
      "## Sources",
    ].map((marker) => md.indexOf(marker));

    expect(order.every((index) => index >= 0)).toBe(true);
    expect(order).toEqual([...order].sort((a, b) => a - b));
    expect(md.match(/Canonical Council conflict/g)).toHaveLength(1);
    expect(md).not.toContain("Legacy duplicate");
  });

  it("falls back to legacy top-level Council conflicts only when answer conflicts are absent", () => {
    const council: RunResult = {
      ...baseResult,
      mode: "council",
      conflicts: [
        {
          id: "conflict-legacy",
          claim_id: "claim-legacy",
          claim: "Legacy Council conflict",
          type: "peer_unresolved",
          claim_scope: "same scope",
          evidence_ids: ["c1", "c2"],
          resolution: "contested",
          reason: "Legacy result compatibility.",
        },
      ],
      members: [{ model: "model-x", answer: baseResult.answer }],
    };

    const md = runResultToMarkdown("Topic", council, []);

    expect(md.match(/Legacy Council conflict/g)).toHaveLength(1);
  });

  it("keeps temporal scope, body, conflict note, freshness note, limitations, process, and sources in contract order", () => {
    const result: RunResult = {
      ...baseResult,
      mode: "deep",
      status: "completed_degraded",
      temporal_coverage: {
        intent: "latest",
        checked_at: 10,
        status: "unknown",
        basis: "none",
        evidence_ids: [],
        reason_codes: ["temporal_coverage_unknown"],
        statement: "时效范围：现有证据无法确认覆盖到当前时点。",
        note_required: true,
      },
      answer: {
        text: "正文结论 [c1]。",
        citations: [],
        claim_map: {},
        coverage: 0.9,
        limitations: "仍有样本限制。",
        confidence_label: "contested",
        conflicts: [
          {
            id: "conflict-1",
            claim_id: "claim-1",
            claim: "指标方向不一致",
            type: "peer_unresolved",
            claim_scope: "same scope",
            evidence_ids: ["local:library_note:n1", "external:web_document:d1"],
            resolution: "contested",
            reason: "同范围证据仍无法裁决。",
          },
        ],
      },
      sections: [{ title: "Method", answer: { text: "过程说明。", citations: [], claim_map: {}, coverage: 1 } }],
    };

    const md = runResultToMarkdown("Topic", result, sources);
    const order = [
      "> 时效范围声明：",
      "正文结论",
      "## 证据冲突附注",
      "## 时效性附注",
      "## Limitations",
      "## Research process",
      "## Sources",
    ].map((marker) => md.indexOf(marker));

    expect(order.every((index) => index >= 0)).toBe(true);
    expect(order).toEqual([...order].sort((a, b) => a - b));
    expect(md).toContain("指标方向不一致");
    expect(md).toContain("temporal_coverage_unknown");
  });
});

describe("slugifyQuery", () => {
  it("produces a filesystem-safe slug", () => {
    expect(slugifyQuery("Hello, World! 2026")).toBe("hello-world-2026");
  });
  it("falls back when empty", () => {
    expect(slugifyQuery("!!!")).toBe("deepresearch");
  });
});
