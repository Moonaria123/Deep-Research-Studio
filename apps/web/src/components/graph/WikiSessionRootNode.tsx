import type { CSSProperties } from "react";
import type { NodeProps } from "@xyflow/react";
import { GraphNodeHandles } from "@/components/graph/GraphNodeHandles";
import { stringAttr, type GraphFlowNode } from "@/components/graph/GraphNode";
import { cn, formatTime } from "@/lib/utils";

const clampStyle = (lines: number): CSSProperties => ({
  display: "-webkit-box",
  WebkitBoxOrient: "vertical",
  WebkitLineClamp: lines,
  overflow: "hidden",
  overflowWrap: "anywhere",
  wordBreak: "break-word",
});

export function WikiSessionRootNode({ data }: NodeProps<GraphFlowNode>) {
  const node = data.graphNode;
  const related = stringAttr(node, "wiki_related") === "true";
  const collapsedChildren = Number.parseInt(stringAttr(node, "collapsed_children") || "0", 10);
  const topicCount = typeof node.attrs?.topic_count === "number" ? node.attrs.topic_count : null;
  const refreshedAt = typeof node.attrs?.last_refreshed_at === "number" ? node.attrs.last_refreshed_at : 0;
  const generationState = stringAttr(node, "generation_state");
  const stale = node.attrs?.stale === true;

  return (
    <div
      data-testid={`graph-node-${node.id}`}
      className={cn(
        "relative w-[300px] rounded-md border-2 px-3 py-2 text-left shadow-sm transition-colors",
        data.selected || data.highlighted ? "shadow-md" : "",
      )}
      style={{
        background: related ? "#f8fafc" : "#eef2ff",
        borderColor: related ? "rgba(100, 116, 139, 0.55)" : "#4f46e5",
        color: related ? "#475569" : "#312e81",
      }}
    >
      <GraphNodeHandles />
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="rounded-sm border border-current/20 bg-white/60 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-normal">
          {related ? "Related Session" : "Session"}
        </span>
        {collapsedChildren > 0 ? (
          <span className="rounded-sm bg-white/65 px-1.5 py-0.5 text-[10px] font-semibold">+{collapsedChildren}</span>
        ) : null}
      </div>
      <div data-testid={`graph-node-title-${node.id}`} className="text-sm font-semibold leading-snug" style={clampStyle(2)}>
        {node.title || stringAttr(node, "session_id") || "Untitled session"}
      </div>
      {node.summary ? (
        <div className="mt-1 text-[11px] leading-snug opacity-80" style={clampStyle(3)}>
          {node.summary}
        </div>
      ) : null}
      <div className="mt-2 flex flex-wrap items-center gap-1 text-[10px] opacity-80">
        <span className="rounded-sm bg-white/65 px-1.5 py-0.5">{topicCount === null ? "Topic count unknown" : `${topicCount} topics`}</span>
        {generationState ? (
          <span className="rounded-sm border border-current/20 bg-white/65 px-1.5 py-0.5 font-semibold">
            {generationState[0].toUpperCase()}{generationState.slice(1)}
          </span>
        ) : null}
        {stale ? <span className="rounded-sm bg-amber-100 px-1.5 py-0.5 font-semibold text-amber-700">Stale</span> : null}
      </div>
      {refreshedAt > 0 ? <div className="mt-1 text-[10px] opacity-70">Updated {formatTime(refreshedAt)}</div> : null}
    </div>
  );
}
