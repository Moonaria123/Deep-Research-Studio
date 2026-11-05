import type { GroundedAnswer } from "./types";

export interface CouncilSections {
  agreement: string[];
  disagreement: string[];
  uniqueFindings: string[];
  synthesis: string;
}

const HEADING_RE = /^##\s+(.+?)\s*$/;
const BULLET_RE = /^(?:[-*•]|\d+[.)])\s+(.*)$/;

/** Collapse a heading to a comparison key, e.g. "Unique findings" -> "uniquefindings". */
function normalizeHeading(heading: string): string {
  return heading.toLowerCase().replace(/[^a-z]/g, "");
}

/** Split a markdown body into a map of normalized ## heading -> raw section body. */
function splitSections(text: string): Map<string, string> {
  const out = new Map<string, string>();
  let current: string | null = null;
  let buffer: string[] = [];
  const flush = () => {
    if (current) out.set(current, buffer.join("\n").trim());
    buffer = [];
  };
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(HEADING_RE);
    if (match) {
      flush();
      current = normalizeHeading(match[1]);
    } else if (current) {
      buffer.push(line);
    }
  }
  flush();
  return out;
}

/** Extract list items from a section body, folding wrapped continuation lines. */
function bullets(body: string): string[] {
  if (!body) return [];
  const items: string[] = [];
  for (const raw of body.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const match = line.match(BULLET_RE);
    if (match) {
      items.push(match[1].trim());
    } else if (items.length) {
      items[items.length - 1] += ` ${line}`;
    } else {
      // Prose line before any bullet (rare) — keep it as its own row.
      items.push(line);
    }
  }
  return items.filter(Boolean);
}

/**
 * Parse a council consensus answer into matrix sections (Agreement / Disagreement /
 * Unique findings / Synthesis). Returns ``null`` when none of the matrix headings are
 * present so the caller can fall back to rendering the raw consensus Markdown.
 */
export function parseCouncilSections(answer: GroundedAnswer): CouncilSections | null {
  const sections = splitSections(answer.text ?? "");
  const hasMatrix = ["agreement", "disagreement", "uniquefindings"].some((key) => sections.has(key));
  if (!hasMatrix) return null;
  return {
    agreement: bullets(sections.get("agreement") ?? ""),
    disagreement: bullets(sections.get("disagreement") ?? ""),
    uniqueFindings: bullets(sections.get("uniquefindings") ?? ""),
    synthesis: (sections.get("synthesis") ?? "").trim(),
  };
}
