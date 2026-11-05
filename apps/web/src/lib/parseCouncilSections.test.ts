import { describe, expect, it } from "vitest";
import { parseCouncilSections } from "./parseCouncilSections";
import type { GroundedAnswer } from "./types";

const answer = (text: string): GroundedAnswer => ({
  text,
  citations: [],
  claim_map: {},
  coverage: 0.9,
});

describe("parseCouncilSections", () => {
  it("returns null when no matrix headings are present", () => {
    expect(parseCouncilSections(answer("Just a plain paragraph with no headings."))).toBeNull();
    expect(parseCouncilSections(answer("## Conclusion\nSome text."))).toBeNull();
  });

  it("splits bulleted Agreement/Disagreement/Unique findings/Synthesis sections", () => {
    const text = [
      "## Agreement",
      "- Moon gravity dominates [c1]",
      "- Tides are semidiurnal [c2]",
      "",
      "## Disagreement",
      "- Solar weighting differs [c2]",
      "",
      "## Unique findings",
      "- One member noted resonance basins [c1]",
      "",
      "## Synthesis",
      "Lunar gravity dominates while the Sun modulates spring tides [c1] [c2].",
    ].join("\n");
    const sections = parseCouncilSections(answer(text));
    expect(sections).not.toBeNull();
    expect(sections!.agreement).toEqual(["Moon gravity dominates [c1]", "Tides are semidiurnal [c2]"]);
    expect(sections!.disagreement).toEqual(["Solar weighting differs [c2]"]);
    expect(sections!.uniqueFindings).toEqual(["One member noted resonance basins [c1]"]);
    expect(sections!.synthesis).toContain("Lunar gravity dominates");
  });

  it("treats prose lines under a heading as rows and folds wrapped continuations", () => {
    const text = [
      "## Agreement",
      "The Moon's gravity is the dominant tidal driver",
      "across all members [c1].",
      "## Disagreement",
      "No material disagreement.",
    ].join("\n");
    const sections = parseCouncilSections(answer(text));
    expect(sections!.agreement).toEqual(["The Moon's gravity is the dominant tidal driver across all members [c1]."]);
    expect(sections!.disagreement).toEqual(["No material disagreement."]);
    expect(sections!.uniqueFindings).toEqual([]);
  });
});
