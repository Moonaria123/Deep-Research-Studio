import { Link2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export function SessionMapControls({
  showCrossLinks,
  onToggleCrossLinks,
}: {
  showCrossLinks: boolean;
  onToggleCrossLinks: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2" aria-label="Session Map controls">
      <Button variant="outline" size="sm" aria-pressed={showCrossLinks} onClick={onToggleCrossLinks}>
        <Link2 className="h-4 w-4" />
        <span className="max-[520px]:sr-only">{showCrossLinks ? "Hide cross-session links" : "Show cross-session links"}</span>
      </Button>
    </div>
  );
}
