import { describe, expect, it } from "vitest";
import {
  buildClaimLookup,
  citationIdsToSourceIndexes,
  normalizeClaimKey,
  resolveClaimForSentence,
  sameIdSet,
  sentenceRanges,
  splitSentences,
} from "./claimMap";
import type { Citation } from "@/lib/types";

const cite = (id: string): Citation => ({ id, document_id: id, url: "", title: "", snippet: "", score: 1 });

describe("splitSentences", () => {
  it("splits on terminal punctuation and newlines like the backend", () => {
    expect(splitSentences("One fact. Two facts! Three?\nFour.")).toEqual([
      "One fact.",
      "Two facts!",
      "Three?",
      "Four.",
    ]);
  });

  it("handles CJK terminators", () => {
    expect(splitSentences("第一句。第二句！？第三句；第四句")).toEqual([
      "第一句。",
      "第二句！？",
      "第三句；",
      "第四句",
    ]);
  });

  it("protects decimals, URLs and common abbreviations", () => {
    expect(
      splitSentences(
        "Value 3.14 was reported by Dr. Chen. See https://example.com/research/v1.2 for details. Next claim.",
      ),
    ).toEqual([
      "Value 3.14 was reported by Dr. Chen.",
      "See https://example.com/research/v1.2 for details.",
      "Next claim.",
    ]);
  });

  it("protects URL query punctuation", () => {
    expect(
      splitSentences("See https://example.com/search?q=alpha;lang=en for details. Next claim."),
    ).toEqual(["See https://example.com/search?q=alpha;lang=en for details.", "Next claim."]);
  });

  it("returns empty for blank input", () => {
    expect(splitSentences("   ")).toEqual([]);
  });
});

describe("sentenceRanges", () => {
  it("fully partitions the text with no gaps", () => {
    const text = "A fact. Another one.";
    const ranges = sentenceRanges(text);
    expect(ranges.map(([s, e]) => text.slice(s, e)).join("")).toBe(text);
    expect(ranges.map(([s, e]) => text.slice(s, e).trim())).toEqual(["A fact.", "Another one."]);
  });

  it("returns a single range when there is no boundary", () => {
    expect(sentenceRanges("no terminator")).toEqual([[0, "no terminator".length]]);
  });

  it("keeps CJK punctuation clusters in the preceding range", () => {
    const text = "第一句！？第二句；第三句";
    const ranges = sentenceRanges(text);
    expect(ranges.map(([s, e]) => text.slice(s, e))).toEqual(["第一句！？", "第二句；", "第三句"]);
    expect(ranges.map(([s, e]) => text.slice(s, e)).join("")).toBe(text);
  });
});

describe("normalizeClaimKey", () => {
  it("strips citation markers, markdown and casing", () => {
    expect(normalizeClaimKey("**Solar** is _renewable_ [c1].")).toBe("solar is renewable .");
  });

  it("strips rendered chip links", () => {
    expect(normalizeClaimKey("Solar is renewable [1](#source-1).")).toBe("solar is renewable .");
  });
});

describe("resolveClaimForSentence", () => {
  const lookup = buildClaimLookup({
    "Solar is renewable [c1].": ["c1"],
    "Wind power is clean [c2][c3].": ["c2", "c3"],
  });

  it("matches a sentence exactly after normalization", () => {
    expect(resolveClaimForSentence("Solar is renewable [c1].", lookup)).toEqual(["c1"]);
  });

  it("matches via substring fallback when chip digits leak in", () => {
    // Rendered text replaces [c2] with the chip number 2.
    expect(resolveClaimForSentence("Wind power is clean 2 3.", lookup)).toEqual(["c2", "c3"]);
  });

  it("returns null for an unmatched sentence", () => {
    expect(resolveClaimForSentence("Totally unrelated claim.", lookup)).toBeNull();
  });

  it("ignores empty claim_map entries", () => {
    const empty = buildClaimLookup({ "x.": [] });
    expect(empty.size).toBe(0);
  });

  it("uses CJK n-grams when rendered citation digits prevent an exact match", () => {
    const cjk = buildClaimLookup({ "电池容量支持长距离续航。": ["c1"] });
    expect(resolveClaimForSentence("电池容量支持长距离续航 1。", cjk)).toEqual(["c1"]);
  });
});

describe("citationIdsToSourceIndexes", () => {
  const citations = [cite("c1"), cite("c2"), cite("c3")];

  it("maps cN to 1-based source indexes", () => {
    expect(citationIdsToSourceIndexes(["c1", "c3"], citations)).toEqual([1, 3]);
  });

  it("drops ids absent from citations and de-duplicates", () => {
    expect(citationIdsToSourceIndexes(["c1", "c1", "c9"], citations)).toEqual([1]);
  });
});

describe("sameIdSet", () => {
  it("compares regardless of order", () => {
    expect(sameIdSet(["c1", "c2"], ["c2", "c1"])).toBe(true);
    expect(sameIdSet(["c1"], ["c1", "c2"])).toBe(false);
  });
});
