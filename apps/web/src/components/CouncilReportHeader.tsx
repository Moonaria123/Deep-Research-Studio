import { Users } from "lucide-react";
import type { CouncilMember } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

const STRENGTH_COPY: Record<string, string> = {
  strong: "Strong consensus",
  mixed: "Mixed consensus",
  divergent: "Divergent views",
};

/**
 * A compact header for the council report: how many models deliberated, which models
 * took part, and the qualitative consensus strength.
 */
export function CouncilReportHeader({
  members,
  strength,
}: {
  members: CouncilMember[];
  strength?: string;
}) {
  if (!members.length) return null;
  const strengthLabel = strength ? STRENGTH_COPY[strength] ?? strength : undefined;
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-bg-subtle/60 px-3 py-2 text-xs text-text-muted">
      <span className="flex items-center gap-1.5 font-medium text-text-body">
        <Users className="h-3.5 w-3.5" aria-hidden="true" />
        {members.length}-model council
      </span>
      <span className="flex flex-wrap items-center gap-1.5">
        {members.map((m, i) => (
          <Badge key={`${m.model}-${i}`} variant="default">
            {m.model}
          </Badge>
        ))}
      </span>
      {strengthLabel && (
        <span className="ml-auto">
          <Badge variant="accent">{strengthLabel}</Badge>
        </span>
      )}
    </div>
  );
}
