import { Handle, Position } from "@xyflow/react";

/** Invisible handles required for @xyflow/react v12 edge rendering on custom nodes. */
export function GraphNodeHandles() {
  return (
    <>
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2 !w-2 !border !border-slate-400 !bg-white !opacity-0"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!h-2 !w-2 !border !border-slate-400 !bg-white !opacity-0"
      />
    </>
  );
}
