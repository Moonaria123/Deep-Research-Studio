import type { NodeProps } from "@xyflow/react";
import { Database, FileSearch } from "lucide-react";
import { GraphNodeHandles } from "@/components/graph/GraphNodeHandles";
import { cn } from "@/lib/utils";
import { type GraphFlowNode } from "@/components/graph/GraphNode";

export function HubNode({ data }: NodeProps<GraphFlowNode>) {
  const kind = data.hubKind ?? "source";
  const count = data.hubCount ?? 0;
  const expanded = !!data.expanded;
  const isMemory = kind === "memory";
  const Icon = isMemory ? Database : FileSearch;

  return (
    <div
      data-testid={`graph-node-${data.graphNode.id}`}
      className={cn(
        "relative grid h-[92px] w-[132px] place-items-center rounded-full border-2 px-3 text-center shadow-sm transition-colors",
        isMemory ? "border-rose-500 bg-rose-50 text-rose-950" : "border-cyan-600 bg-cyan-50 text-cyan-950",
        expanded ? "ring-2 ring-accent/45" : "",
      )}
    >
      <GraphNodeHandles />
      <div>
        <Icon className="mx-auto mb-1 h-4 w-4" aria-hidden />
        <div className="text-xs font-bold">{isMemory ? "Memories" : "Sources"}</div>
        <div className="text-lg font-bold leading-tight">{count}</div>
        <div className="text-[10px] font-medium opacity-70">{expanded ? "Collapse" : "Expand"}</div>
      </div>
    </div>
  );
}
