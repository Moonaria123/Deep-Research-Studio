import type { CSSProperties } from "react";
import type { NodeProps } from "@xyflow/react";
import { GraphNodeHandles } from "@/components/graph/GraphNodeHandles";
import { formatTime } from "@/lib/utils";
import { type GraphFlowNode } from "@/components/graph/GraphNode";

const clampStyle = (lines: number): CSSProperties => ({
  display: "-webkit-box",
  WebkitBoxOrient: "vertical",
  WebkitLineClamp: lines,
  overflow: "hidden",
  overflowWrap: "anywhere",
  wordBreak: "break-word",
});

export function SessionNode({ data }: NodeProps<GraphFlowNode>) {
  const node = data.graphNode;
  const counts = data.counts ?? { reports: 0, notes: 0, sources: 0, memories: 0 };
  const expanded = !!data.expanded;
  const extended = node as typeof node & {
    assistant_excerpt?: string;
    assistant_summary?: string;
    last_assistant_message_id?: string;
    last_assistant_at?: string;
  };
  const summary = String(extended.assistant_excerpt || extended.assistant_summary || node.attrs?.assistant_excerpt || node.attrs?.assistant_summary || node.summary || "");
  const messageId = String(extended.last_assistant_message_id || node.attrs?.assistant_message_id || node.attrs?.last_assistant_message_id || "");
  const assistantAt = Number(extended.last_assistant_at || node.attrs?.assistant_created_at || node.attrs?.last_assistant_at || 0);

  return (
    <div
      data-testid={`graph-node-${node.id}`}
      className="relative w-[320px] rounded-md border-2 border-indigo-500 bg-indigo-50 px-4 py-3 text-left text-indigo-950 shadow-sm transition-colors"
    >
      <GraphNodeHandles />
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="rounded-sm border border-indigo-700/20 bg-white/65 px-1.5 py-0.5 text-[10px] font-semibold uppercase">
          Session
        </span>
        <span className="text-[10px] font-semibold text-indigo-800">{expanded ? "Expanded" : "Collapsed"}</span>
      </div>
      <div data-testid={`graph-node-title-${node.id}`} className="text-sm font-semibold leading-snug" style={clampStyle(2)}>
        {node.title || node.key}
      </div>
      <p className="mt-2 text-xs leading-snug text-indigo-900/80" style={clampStyle(expanded ? 6 : 3)}>
        {summary || "No assistant response yet."}
      </p>
      {expanded ? (
        <div className="mt-2 space-y-1 text-[11px] text-indigo-900/70">
          {messageId ? <div>Message {messageId}</div> : null}
          {assistantAt ? <div>{formatTime(assistantAt)}</div> : null}
        </div>
      ) : null}
      <div className="mt-3 grid grid-cols-4 gap-1 text-center text-[10px] font-semibold text-indigo-900/80">
        <span>{counts.reports} reports</span>
        <span>{counts.notes} notes</span>
        <span>{counts.sources} sources</span>
        <span>{counts.memories} memory</span>
      </div>
    </div>
  );
}
