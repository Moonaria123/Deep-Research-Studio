import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

const LABELS: Record<string, string> = {
  coverage: "Coverage",
  agreement: "Agreement",
  citation_quality: "Citation quality",
  validator: "Validator",
};

// Calm framing: low cross-model agreement means "members weighed it differently",
// which is informative — not that the finding is weak.
const STRENGTH_COPY: Record<string, string> = {
  strong: "Strong agreement",
  mixed: "Mixed views",
  divergent: "Divergent views",
};

/** Consensus strength with a weighted breakdown bar set. */
export function ConfidenceMeter({
  value,
  breakdown,
  strength,
}: {
  value: number;
  breakdown?: Record<string, number>;
  strength?: string;
}) {
  const pct = Math.round(value * 100);
  const strengthCopy = strength ? STRENGTH_COPY[strength] : undefined;
  return (
    <Card data-anim>
      <CardHeader>
        <CardTitle className="flex items-baseline justify-between text-sm">
          <span>Consensus strength{strengthCopy ? ` · ${strengthCopy}` : ""}</span>
          <span className="text-2xl font-bold text-accent">{pct}%</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="h-2 w-full overflow-hidden rounded-full bg-surface-muted">
          <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${pct}%` }} />
        </div>
        {breakdown && (
          <ul className="space-y-2">
            {Object.entries(breakdown).map(([key, v]) => (
              <li key={key} className="flex items-center gap-3 text-xs">
                <span className="w-28 shrink-0 text-text-muted">{LABELS[key] ?? key}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-muted">
                  <div className="h-full rounded-full bg-accent/70" style={{ width: `${Math.round(v * 100)}%` }} />
                </div>
                <span className="w-9 text-right tabular-nums text-text-faint">{Math.round(v * 100)}%</span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
