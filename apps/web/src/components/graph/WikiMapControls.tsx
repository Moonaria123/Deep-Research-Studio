import { Link2, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

export function WikiMapControls({
  showCrossLinks,
  onToggleCrossLinks,
  onReset,
}: {
  showCrossLinks: boolean;
  onToggleCrossLinks: () => void;
  onReset: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2" aria-label="Wiki Map controls">
      <Button variant="outline" size="sm" aria-pressed={showCrossLinks} onClick={onToggleCrossLinks}>
        <Link2 className="h-4 w-4" />
        <span className="max-[520px]:sr-only">{showCrossLinks ? "Hide cross-session links" : "Show cross-session links"}</span>
      </Button>
      <Button variant="outline" size="sm" onClick={onReset}>
        <RotateCcw className="h-4 w-4" />
        <span className="max-[520px]:sr-only">Reset Wiki</span>
      </Button>
    </div>
  );
}
