import type { Citation } from "@/lib/types";

// Keep this scanner aligned with backend ``_split_claims``. A scanner is used instead
// of look-behind regexes so punctuation remains in the claim while decimal, URL, and
// common-abbreviation periods are not mistaken for sentence boundaries.
const TERMINAL_PUNCTUATION = new Set(".!?;。！？；".split(""));
const COMMON_ABBREVIATIONS = [
  "e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "prof.",
  "sr.", "jr.", "vs.", "etc.", "u.s.", "u.k.",
];

function isAsciiAlphaNumeric(value: string): boolean {
  return /^[a-z0-9]$/i.test(value);
}

function isProtectedPeriod(text: string, index: number): boolean {
  const previous = index > 0 ? text[index - 1] : "";
  const following = index + 1 < text.length ? text[index + 1] : "";
  if (/\d/.test(previous) && /\d/.test(following)) return true;
  const prefix = text.slice(Math.max(0, index - 8), index + 1).toLowerCase();
  if (COMMON_ABBREVIATIONS.some((abbreviation) => prefix.endsWith(abbreviation))) return true;
  let tokenStart = index;
  while (tokenStart > 0 && !/\s/.test(text[tokenStart - 1])) tokenStart -= 1;
  const tokenPrefix = text.slice(tokenStart, index);
  if (following && !/\s/.test(following)) {
    if (tokenPrefix.includes("://") || tokenPrefix.toLowerCase().startsWith("www.")) return true;
    if (isAsciiAlphaNumeric(previous) && isAsciiAlphaNumeric(following)) return true;
  }
  return false;
}

function isInsideUrlToken(text: string, index: number): boolean {
  const following = index + 1 < text.length ? text[index + 1] : "";
  if (!following || /\s/.test(following)) return false;
  let tokenStart = index;
  while (tokenStart > 0 && !/\s/.test(text[tokenStart - 1])) tokenStart -= 1;
  const prefix = text.slice(tokenStart, index).toLowerCase();
  return prefix.includes("://") || prefix.startsWith("www.");
}

/** Split prose into trimmed, non-empty sentences using the backend boundary rule. */
export function splitSentences(text: string): string[] {
  return sentenceRanges(text)
    .map(([start, end]) => text.slice(start, end).trim())
    .filter((sentence) => sentence.length > 0);
}

/**
 * Partition text into [start, end) ranges that fully cover the input. Each range
 * spans one sentence plus its trailing boundary whitespace, so concatenating the
 * slices reproduces the original string with nothing dropped.
 */
export function sentenceRanges(text: string): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  let start = 0;
  let index = 0;
  while (index < text.length) {
    const char = text[index];
    if (char === "\r" || char === "\n") {
      let end = index + 1;
      while (end < text.length && (text[end] === "\r" || text[end] === "\n")) end += 1;
      ranges.push([start, end]);
      start = end;
      index = end;
      continue;
    }
    if (
      !TERMINAL_PUNCTUATION.has(char) ||
      isInsideUrlToken(text, index) ||
      (char === "." && isProtectedPeriod(text, index))
    ) {
      index += 1;
      continue;
    }
    let end = index + 1;
    while (end < text.length && TERMINAL_PUNCTUATION.has(text[end])) end += 1;
    while (end < text.length && /[\t ]/.test(text[end])) end += 1;
    ranges.push([start, end]);
    start = end;
    index = end;
  }
  if (start < text.length) ranges.push([start, text.length]);
  if (ranges.length === 0) ranges.push([0, text.length]);
  return ranges;
}

/**
 * Normalize a sentence into a comparable key: strip inline citation markers,
 * already-linked chips, markdown emphasis/code/heading punctuation, collapse
 * whitespace and lowercase. Makes client sentences comparable to backend keys.
 */
export function normalizeClaimKey(sentence: string): string {
  return sentence
    .replace(/\[c\d+\]/gi, " ") // raw [cN] citation markers
    .replace(/\[(\d+)\]\([^)]*\)/g, " ") // rendered chips [3](#source-3 "...")
    .replace(/[*_`#>~]+/g, " ") // markdown emphasis / code / heading / quote markers
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

/** Build a normalized lookup from the backend claim_map (sentence -> citation ids). */
export function buildClaimLookup(claimMap: Record<string, string[]> | undefined): Map<string, string[]> {
  const lookup = new Map<string, string[]>();
  if (!claimMap) return lookup;
  for (const [key, ids] of Object.entries(claimMap)) {
    const norm = normalizeClaimKey(key);
    if (norm && ids.length) lookup.set(norm, ids);
  }
  return lookup;
}

/** Word tokens for fuzzy matching: drops pure-punctuation/digit tokens (chip numbers). */
function wordTokens(normalized: string): Set<string> {
  const tokens = new Set<string>();
  for (const token of normalized.match(/[a-z0-9]+/g) ?? []) {
    if (token.length > 2 && !/^\d+$/.test(token)) tokens.add(token);
  }
  for (const run of normalized.match(/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/g) ?? []) {
    for (let index = 0; index < run.length - 1; index += 1) tokens.add(run.slice(index, index + 2));
  }
  return tokens;
}

/**
 * Resolve the citation ids backing a sentence. Tries an exact normalized match,
 * then a token-overlap fallback that tolerates citation-chip digits leaking into
 * the rendered text (the chip renders the number, while the key strips ``[cN]``).
 * A key matches when >= 80% of its content words appear in the sentence.
 */
export function resolveClaimForSentence(sentence: string, lookup: Map<string, string[]>): string[] | null {
  const norm = normalizeClaimKey(sentence);
  if (!norm) return null;
  const exact = lookup.get(norm);
  if (exact) return exact;
  const sentTokens = wordTokens(norm);
  if (sentTokens.size === 0) return null;
  let best: string[] | null = null;
  let bestScore = 0.8; // minimum coverage threshold
  for (const [key, ids] of lookup) {
    const keyTokens = wordTokens(key);
    if (keyTokens.size === 0) continue;
    let inter = 0;
    for (const t of keyTokens) if (sentTokens.has(t)) inter += 1;
    const score = inter / keyTokens.size;
    if (score >= bestScore) {
      best = ids;
      bestScore = score;
    }
  }
  return best;
}

/**
 * Map citation ids (``c3``) to 1-based SourcesRail indexes, matching the existing
 * ``[cN] -> #source-N`` convention used by AnswerBody. Filters ids absent from the
 * provided citations and de-duplicates.
 */
export function citationIdsToSourceIndexes(ids: string[], citations: Citation[]): number[] {
  const valid = new Set(citations.map((c) => c.id));
  const out = new Set<number>();
  for (const id of ids) {
    if (valid.size > 0 && !valid.has(id)) continue;
    const n = Number(id.replace(/\D/g, ""));
    if (Number.isFinite(n) && n > 0) out.add(n);
  }
  return [...out];
}

/** True when two id lists contain the same set of ids (order-independent). */
export function sameIdSet(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const set = new Set(a);
  return b.every((id) => set.has(id));
}
