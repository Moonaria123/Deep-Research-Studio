import type { EvidenceMatrixRow } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface DeepEvidenceMatrixProps {
  rows: EvidenceMatrixRow[];
}

function statusLabel(status: string): string {
  return status.replace(/_/g, " ");
}

export function DeepEvidenceMatrix({ rows }: DeepEvidenceMatrixProps) {
  if (!rows.length) return null;

  return (
    <Card data-anim>
      <CardHeader>
        <CardTitle className="text-sm">Evidence matrix</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {rows.map((row, index) => (
          <div key={`${row.subtask}-${index}`} className="rounded-md border border-border bg-bg-subtle/60 p-3">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="font-medium text-text-body">{row.subtask}</p>
                <p className="mt-1 text-sm leading-6 text-text-muted">{row.notes}</p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Badge variant={row.status === "gap_closed" ? "accent" : "default"}>{statusLabel(row.status)}</Badge>
                <span className="rounded bg-surface px-2 py-1 text-xs text-text-muted">{row.source_count}</span>
              </div>
            </div>
            {row.gaps.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {row.gaps.map((gap) => (
                  <Badge key={gap} variant="default">
                    {gap}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
