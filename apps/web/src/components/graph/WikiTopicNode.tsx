import type { CSSProperties } from "react";
import type { NodeProps } from "@xyflow/react";
import { GraphNodeHandles } from "@/components/graph/GraphNodeHandles";
import { stringAttr, type GraphFlowNode } from "@/components/graph/GraphNode";
import { cn } from "@/lib/utils";

const TOPIC_COLORS = [
  { bg: "#f8fafc", border: "#475569", text: "#1f2937" },
  { bg: "#eef2ff", border: "#4f46e5", text: "#312e81" },
  { bg: "#ecfeff", border: "#0891b2", text: "#164e63" },
  { bg: "#f0fdf4", border: "#16a34a", text: "#14532d" },
  { bg: "#fff7ed", border: "#ea580c", text: "#7c2d12" },
  { bg: "#fdf2f8", border: "#db2777", text: "#831843" },
];

const clampStyle = (lines: number): CSSProperties => ({
  display: "-webkit-box",
  WebkitBoxOrient: "vertical",
  WebkitLineClamp: lines,
  overflow: "hidden",
  overflowWrap: "anywhere",
  wordBreak: "break-word",
});

function topicColors(value: string) {
  const index = Number.parseInt(value || "0", 10);
  return TOPIC_COLORS[Math.abs(Number.isFinite(index) ? index : 0) % TOPIC_COLORS.length];
}

function evidenceRefs(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => !!item && typeof item === "object") : [];
}

function evidenceLabel(refs: Array<Record<string, unknown>>): string {
  if (refs.length === 0) return "";
  const counts = refs.reduce<Record<string, number>>((acc, ref) => {
    const type = typeof ref.type === "string" && ref.type ? ref.type : "evidence";
    acc[type] = (acc[type] ?? 0) + 1;
    return acc;
  }, {});
  const [type, count] = Object.entries(counts).sort((left, right) => right[1] - left[1])[0];
  return `${count} ${type}${count === 1 ? "" : "s"}`;
}

export function WikiTopicNode(props: NodeProps<GraphFlowNode>) {
  const node = props.data.graphNode;
  const colors = topicColors(stringAttr(node, "color_index"));
  const related = stringAttr(node, "wiki_related") === "true";
  const collapsedChildren = Number.parseInt(stringAttr(node, "collapsed_children") || "0", 10);
  const refs = evidenceRefs(node.attrs?.evidence_refs);
  const quality = stringAttr(node, "quality");
  const evidence = evidenceLabel(refs);

  return (
    <div
      data-testid={`graph-node-${node.id}`}
      className={cn(
        "relative w-[248px] rounded-md border-2 px-3 py-2 text-left shadow-sm transition-colors",
        props.data.selected || props.data.highlighted ? "shadow-md" : "",
      )}
      style={{
        background: related ? "#f8fafc" : colors.bg,
        borderColor: props.data.selected || props.data.highlighted ? colors.border : related ? "rgba(100, 116, 139, 0.45)" : "rgba(100, 116, 139, 0.35)",
        color: related ? "#475569" : colors.text,
      }}
    >
      <GraphNodeHandles />
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="rounded-sm border border-current/20 bg-white/55 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-normal">
          {related ? "Related Topic" : "Topic"}
        </span>
        {collapsedChildren > 0 ? (
          <span className="rounded-sm bg-white/65 px-1.5 py-0.5 text-[10px] font-semibold">+{collapsedChildren}</span>
        ) : null}
      </div>
      {evidence || quality ? (
        <div className="mb-1 flex flex-wrap gap-1">
          {evidence ? <span className="rounded-sm bg-white/65 px-1.5 py-0.5 text-[10px] font-medium">{evidence}</span> : null}
          {quality ? <span className="rounded-sm bg-white/55 px-1.5 py-0.5 text-[10px] font-medium">{quality}</span> : null}
        </div>
      ) : null}
      <div data-testid={`graph-node-title-${node.id}`} className="text-xs font-semibold leading-snug" style={clampStyle(2)}>
        {node.title || "Untitled topic"}
      </div>
      {node.summary ? (
        <div className="mt-1 text-[11px] leading-snug opacity-80" style={clampStyle(2)}>
          {node.summary}
        </div>
      ) : null}
    </div>
  );
}
