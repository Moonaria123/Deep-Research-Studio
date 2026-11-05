import type { CSSProperties } from "react";
import type { NodeProps } from "@xyflow/react";
import { StickyNote } from "lucide-react";
import { GraphNodeHandles } from "@/components/graph/GraphNodeHandles";
import { type GraphFlowNode, stringAttr } from "@/components/graph/GraphNode";

const clampStyle = (lines: number): CSSProperties => ({
  display: "-webkit-box",
  WebkitBoxOrient: "vertical",
  WebkitLineClamp: lines,
  overflow: "hidden",
  overflowWrap: "anywhere",
  wordBreak: "break-word",
});

function looksLikeRawTitle(node: GraphFlowNode["data"]["graphNode"], title: string): boolean {
  if (!title) return false;
  const noteId = stringAttr(node, "note_id");
  return title === node.key || title === noteId || /^(note|mem)[:\s_-]/i.test(title);
}

function displayTitle(node: GraphFlowNode["data"]["graphNode"]): string {
  if (node.summary) return node.summary;
  if (node.title && !looksLikeRawTitle(node, node.title)) return node.title;
  return "No note summary yet.";
}

export function NoteMemberNode({ data }: NodeProps<GraphFlowNode>) {
  const node = data.graphNode;
  const expanded = !!data.expanded;
  const namespace = stringAttr(node, "namespace");
  const scope = stringAttr(node, "scope_key");
  const source = stringAttr(node, "source");
  const excerpt = stringAttr(node, "excerpt") || stringAttr(node, "content_excerpt") || node.summary || "";
  const provenance = [namespace, scope, source].filter(Boolean).join(" / ");

  return (
    <div
      data-testid={`graph-node-${node.id}`}
      className="relative w-[248px] rounded-md border-2 border-orange-600 bg-orange-50 px-3 py-2 text-left text-orange-950 shadow-sm transition-colors"
    >
      <GraphNodeHandles />
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-1 rounded-sm border border-orange-700/20 bg-white/65 px-1.5 py-0.5 text-[10px] font-semibold uppercase">
          <StickyNote className="h-3 w-3" aria-hidden />
          Note
        </span>
        {namespace ? <span className="min-w-0 truncate text-[10px] font-medium opacity-75">{namespace}</span> : null}
      </div>
      <div
        data-testid={`graph-node-title-${node.id}`}
        className="text-xs font-semibold leading-snug"
        style={clampStyle(expanded ? 4 : 2)}
      >
        {displayTitle(node)}
      </div>
      {expanded ? (
        <div className="mt-2 space-y-1 text-[11px] leading-snug text-orange-900/75">
          {excerpt ? <p style={clampStyle(5)}>{excerpt}</p> : <p>No note excerpt available.</p>}
          {provenance ? <p className="font-medium">Provenance: {provenance}</p> : null}
          <p className="font-semibold text-orange-800">Open note</p>
        </div>
      ) : null}
    </div>
  );
}
