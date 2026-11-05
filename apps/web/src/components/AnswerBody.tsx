import {
  Children,
  Fragment,
  isValidElement,
  useMemo,
  type ComponentPropsWithoutRef,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "@/lib/types";
import { buildClaimLookup, resolveClaimForSentence, sentenceRanges } from "@/lib/claimMap";

const CITATION_RE = /\[(c\d+)\]/g;
// Fenced (```...```) and inline (`...`) code regions, captured so String.split
// keeps them; their content must never be rewritten by the citation transform.
const CODE_RE = /(```[\s\S]*?```|`[^`]*`)/g;

export interface ClaimHighlightProps {
  /** Backend claim_map: sentence text -> supporting citation ids. */
  claimMap?: Record<string, string[]>;
  /** Currently focused citation ids; a sentence lights up when it intersects. */
  activeIds?: string[];
  /** Hover preview: ids on enter, null on leave. */
  onClaimFocus?: (ids: string[] | null) => void;
  /** Click to pin/unpin a sentence's ids. */
  onClaimPin?: (ids: string[]) => void;
}

/** Flatten any React node tree into its plain text content. */
function nodeText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (isValidElement(node)) return nodeText((node.props as { children?: ReactNode }).children);
  return "";
}

interface SentenceGroup {
  text: string;
  nodes: ReactNode[];
}

/**
 * Group a paragraph's rendered children into per-sentence buckets using the same
 * boundary rule as the backend, splitting string children at internal boundaries
 * while keeping inline elements (citation chips) atomic.
 */
function groupChildrenBySentence(children: ReactNode): SentenceGroup[] {
  const parts = Children.toArray(children);
  const full = parts.map(nodeText).join("");
  const ranges = sentenceRanges(full);
  const groups: SentenceGroup[] = ranges.map(([start, end]) => ({ text: full.slice(start, end), nodes: [] }));

  const rangeIndexAt = (pos: number): number => {
    for (let i = 0; i < ranges.length; i += 1) {
      if (pos < ranges[i][1]) return i;
    }
    return ranges.length - 1;
  };

  let pos = 0;
  let keySeq = 0;
  for (const part of parts) {
    const len = nodeText(part).length;
    if (typeof part === "string") {
      // A string child may straddle several sentences: emit per-range slices.
      let cursor = 0;
      while (cursor < part.length) {
        const idx = rangeIndexAt(pos + cursor);
        const rangeEnd = ranges[idx][1];
        const take = Math.min(part.length - cursor, rangeEnd - (pos + cursor));
        const slice = part.slice(cursor, cursor + Math.max(take, 1));
        groups[idx].nodes.push(slice);
        cursor += Math.max(take, 1);
      }
    } else {
      const idx = rangeIndexAt(pos);
      const node = isValidElement(part) && part.key == null
        ? // give stable keys to chip elements inside their sentence span
          <Fragment key={`cg-${keySeq++}`}>{part}</Fragment>
        : part;
      groups[idx].nodes.push(node);
    }
    pos += len;
  }
  return groups.filter((g) => g.nodes.length > 0);
}

/**
 * Convert inline ``[cN]`` markers into Markdown links pointing at ``#source-N`` so
 * react-markdown renders them as anchored chips, while leaving all other Markdown
 * (headings, lists, bold, tables) and any code spans/blocks intact.
 */
function withCitationLinks(text: string, byId: Map<string, Citation>): string {
  return text
    .split(CODE_RE)
    .map((part, i) =>
      // Odd indices are captured code regions; leave them verbatim.
      i % 2 === 1
        ? part
        : part.replace(CITATION_RE, (_match, id: string) => {
            const num = id.replace(/\D/g, "");
            const title = (byId.get(id)?.title ?? id).replace(/"/g, "'");
            return `[${num}](#source-${num} "${title}")`;
          }),
    )
    .join("");
}

/** Render grounded answer Markdown, turning inline [cN] markers into clickable chips. */
export function AnswerBody({
  text,
  citations,
  claimMap,
  activeIds,
  onClaimFocus,
  onClaimPin,
}: { text: string; citations: Citation[] } & ClaimHighlightProps) {
  const byId = new Map(citations.map((c) => [c.id, c]));
  const source = withCitationLinks(text, byId);
  const lookup = useMemo(() => buildClaimLookup(claimMap), [claimMap]);
  const highlight = lookup.size > 0;

  /**
   * Wrap each matched sentence in a focusable span linked to its citation ids.
   * No-op (returns children unchanged) when there is no claim_map.
   */
  function renderClaims(children: ReactNode): ReactNode {
    if (!highlight) return children;
    const groups = groupChildrenBySentence(children);
    return groups.map((group, i) => {
      const ids = resolveClaimForSentence(group.text, lookup);
      if (!ids || ids.length === 0) {
        return <Fragment key={`s-${i}`}>{group.nodes}</Fragment>;
      }
      const isActive = Boolean(activeIds && ids.some((id) => activeIds.includes(id)));
      return (
        <span
          key={`s-${i}`}
          data-claim
          data-claim-active={isActive ? "true" : undefined}
          data-citation-ids={ids.join(",")}
          role="button"
          tabIndex={0}
          className={`cursor-pointer rounded transition-colors ${
            isActive ? "bg-accent-soft/50" : "hover:bg-accent-soft/25"
          }`}
          onMouseEnter={() => onClaimFocus?.(ids)}
          onMouseLeave={() => onClaimFocus?.(null)}
          onFocus={() => onClaimFocus?.(ids)}
          onBlur={() => onClaimFocus?.(null)}
          onClick={() => onClaimPin?.(ids)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onClaimPin?.(ids);
            }
          }}
        >
          {group.nodes}
        </span>
      );
    });
  }

  return (
    <div className="text-text-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, title, children }: ComponentPropsWithoutRef<"a">) {
            if (typeof href === "string" && href.startsWith("#source-")) {
              return (
                <a
                  href={href}
                  title={title}
                  className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-soft px-1.5 align-baseline text-[11px] font-semibold text-accent hover:brightness-95 focus-ring"
                >
                  {children}
                </a>
              );
            }
            return (
              <a href={href} title={title} target="_blank" rel="noreferrer" className="text-accent underline focus-ring">
                {children}
              </a>
            );
          },
          h1: ({ children }) => <h2 className="mb-2 mt-4 text-lg font-bold text-text">{children}</h2>,
          h2: ({ children }) => <h2 className="mb-2 mt-4 text-base font-semibold text-text">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-1.5 mt-3 text-sm font-semibold text-text">{children}</h3>,
          p: ({ children }) => <p className="mb-3 text-[15px] leading-7 text-text-body">{renderClaims(children)}</p>,
          ul: ({ children }) => (
            <ul className="mb-3 ml-5 list-disc space-y-1 text-[15px] leading-7 text-text-body">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-3 ml-5 list-decimal space-y-1 text-[15px] leading-7 text-text-body">{children}</ol>
          ),
          li: ({ children }) => <li>{renderClaims(children)}</li>,
          strong: ({ children }) => <strong className="font-semibold text-text">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          blockquote: ({ children }) => (
            <blockquote className="mb-3 border-l-2 border-border pl-3 text-text-muted">{children}</blockquote>
          ),
          code: ({ children }) => (
            <code className="rounded bg-bg-subtle px-1 py-0.5 font-mono text-[13px] text-text">{children}</code>
          ),
          table: ({ children }) => (
            <div className="mb-3 overflow-x-auto">
              <table className="w-full border-collapse text-sm">{children}</table>
            </div>
          ),
          th: ({ children }) => <th className="border border-border px-2 py-1 text-left font-semibold">{children}</th>,
          td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
