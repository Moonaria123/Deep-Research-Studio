import type { CSSProperties } from "react";
import type { Node as FlowNode, NodeProps } from "@xyflow/react";
import { GraphNodeHandles } from "@/components/graph/GraphNodeHandles";
import type { KnowledgeGraphNode, KnowledgeGraphNodeType } from "@/lib/types";
import { cn } from "@/lib/utils";

export type GraphNodeData = {
  label: string;
  graphNode: KnowledgeGraphNode;
  selected?: boolean;
  highlighted?: boolean;
  expanded?: boolean;
  hubKind?: "source" | "memory";
  hubCount?: number;
  counts?: { reports: number; notes: number; sources: number; memories: number };
};

export type GraphFlowNode = FlowNode<GraphNodeData>;

export const NODE_TYPE_LABELS: Record<string, string> = {
  session: "Session",
  source: "Source",
  note: "Note",
  report: "Report",
  memory: "Memory",
  sourceHub: "Sources",
  memoryHub: "Memories",
  wikiTopic: "Topic",
  concept: "Concept",
};

export const NODE_COLORS: Record<string, { bg: string; border: string; text: string; minimap: string }> = {
  session: { bg: "#eef2ff", border: "#4f46e5", text: "#312e81", minimap: "#4f46e5" },
  source: { bg: "#ecfeff", border: "#0891b2", text: "#164e63", minimap: "#0891b2" },
  report: { bg: "#f0fdf4", border: "#16a34a", text: "#14532d", minimap: "#16a34a" },
  note: { bg: "#fff7ed", border: "#ea580c", text: "#7c2d12", minimap: "#ea580c" },
  memory: { bg: "#fdf2f8", border: "#db2777", text: "#831843", minimap: "#db2777" },
  sourceHub: { bg: "#f0fdff", border: "#0891b2", text: "#164e63", minimap: "#0891b2" },
  memoryHub: { bg: "#fff1f2", border: "#e11d48", text: "#881337", minimap: "#e11d48" },
  wikiTopic: { bg: "#f8fafc", border: "#475569", text: "#1f2937", minimap: "#475569" },
  concept: { bg: "#f5f3ff", border: "#7c3aed", text: "#4c1d95", minimap: "#7c3aed" },
};

const WIKI_TOPIC_COLORS = [
  { bg: "#f8fafc", border: "#475569", text: "#1f2937", minimap: "#475569" },
  { bg: "#eef2ff", border: "#4f46e5", text: "#312e81", minimap: "#4f46e5" },
  { bg: "#ecfeff", border: "#0891b2", text: "#164e63", minimap: "#0891b2" },
  { bg: "#f0fdf4", border: "#16a34a", text: "#14532d", minimap: "#16a34a" },
  { bg: "#fff7ed", border: "#ea580c", text: "#7c2d12", minimap: "#ea580c" },
  { bg: "#fdf2f8", border: "#db2777", text: "#831843", minimap: "#db2777" },
];

const clampStyle = (lines: number): CSSProperties => ({
  display: "-webkit-box",
  WebkitBoxOrient: "vertical",
  WebkitLineClamp: lines,
  overflow: "hidden",
  overflowWrap: "anywhere",
  wordBreak: "break-word",
});

export function typeLabel(type: KnowledgeGraphNodeType): string {
  return NODE_TYPE_LABELS[type] ?? type;
}

export function colorFor(type: KnowledgeGraphNodeType) {
  return NODE_COLORS[type] ?? { bg: "#f8fafc", border: "#64748b", text: "#334155", minimap: "#64748b" };
}

function colorForNode(node: KnowledgeGraphNode) {
  if (node.type !== "wikiTopic") return colorFor(node.type);
  const value = node.attrs?.color_index;
  const index = typeof value === "string" ? Number.parseInt(value, 10) : typeof value === "number" ? value : 0;
  return WIKI_TOPIC_COLORS[Math.abs(Number.isFinite(index) ? index : 0) % WIKI_TOPIC_COLORS.length];
}

export function stringAttr(node: KnowledgeGraphNode, key: string): string {
  const value = node.attrs?.[key];
  return typeof value === "string" ? value : "";
}

function hostFromUrl(url: string): string {
  if (!url.trim()) return "";
  try {
    return new URL(url).host;
  } catch {
    return url.replace(/^https?:\/\//, "").split("/")[0] ?? "";
  }
}

function primaryText(node: KnowledgeGraphNode): string {
  if (node.type === "source") {
    const domain = hostFromUrl(stringAttr(node, "url")) || node.summary;
    if (node.title && !/^https?:\/\//i.test(node.title)) return node.title;
    return domain || node.key;
  }
  if (node.type === "memory") {
    const memoryId = stringAttr(node, "memory_id");
    const titleLooksRaw = node.title === node.key || node.title === memoryId || /^(memory|mem)[:\s_-]/i.test(node.title);
    return node.summary || (!titleLooksRaw ? node.title : "") || "No memory summary yet.";
  }
  if (node.type === "wikiTopic") return node.title || "Untitled topic";
  return node.title || node.summary || node.key;
}

function secondaryText(node: KnowledgeGraphNode): string {
  if (node.type === "session") return stringAttr(node, "assistant_summary") || node.summary;
  if (node.type === "source") return hostFromUrl(stringAttr(node, "url")) || node.summary;
  if (node.type === "memory") return stringAttr(node, "namespace") || stringAttr(node, "scope_key");
  if (node.type === "wikiTopic") return node.summary || stringAttr(node, "session_title");
  if (node.type === "report") return stringAttr(node, "mode");
  if (node.type === "note") return stringAttr(node, "namespace");
  return node.summary;
}

export function GraphNode({ data }: NodeProps<GraphFlowNode>) {
  const node = data.graphNode;
  const colors = colorForNode(node);
  const selected = !!data.selected;
  const highlighted = !!data.highlighted;
  const emphasis = selected || highlighted;
  const secondary = secondaryText(node);

  return (
    <div
      data-testid={`graph-node-${node.id}`}
      className={cn(
        "relative w-[248px] rounded-md border-2 px-3 py-2 text-left shadow-sm transition-colors",
        emphasis ? "shadow-md" : "",
      )}
      style={{
        background: colors.bg,
        borderColor: emphasis ? colors.border : "rgba(100, 116, 139, 0.35)",
        color: colors.text,
      }}
    >
      <GraphNodeHandles />
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="rounded-sm border border-current/20 bg-white/55 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-normal">
          {typeLabel(node.type)}
        </span>
        {node.type === "memory" && stringAttr(node, "namespace") ? (
          <span className="min-w-0 truncate text-[10px] font-medium opacity-75">{stringAttr(node, "namespace")}</span>
        ) : null}
      </div>
      <div
        data-testid={`graph-node-title-${node.id}`}
        className="text-xs font-semibold leading-snug"
        style={clampStyle(2)}
      >
        {primaryText(node)}
      </div>
      {secondary ? (
        <div className="mt-1 text-[11px] leading-snug opacity-80" style={clampStyle(node.type === "session" ? 3 : 2)}>
          {secondary}
        </div>
      ) : null}
    </div>
  );
}
