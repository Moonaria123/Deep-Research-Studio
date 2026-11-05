import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export type GraphCanvasOverlayItem = {
  id: string;
  node: ReactNode;
  interactive?: boolean;
};

type GraphCanvasOverlayStackProps = {
  items: GraphCanvasOverlayItem[];
};

export function GraphCanvasOverlayStack({ items }: GraphCanvasOverlayStackProps) {
  if (items.length === 0) return null;

  return (
    <div
      aria-label="Graph canvas overlay stack"
      data-testid="graph-canvas-overlay-stack"
      className="pointer-events-none absolute left-4 top-4 z-20 flex max-h-[calc(100%_-_2rem)] w-[calc(100%_-_2rem)] max-w-md flex-col gap-2 overflow-y-auto pr-1"
    >
      {items.map((item) => (
        <div key={item.id} className={cn(item.interactive ? "pointer-events-auto" : "pointer-events-none")}>
          {item.node}
        </div>
      ))}
    </div>
  );
}
