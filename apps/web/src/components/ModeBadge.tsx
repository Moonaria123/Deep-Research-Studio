import type { Mode } from "@/lib/types";
import { isMode, MODE_ARIA_LABEL, MODE_ICON, MODE_SHORT_LABEL } from "@/lib/mode";
import { Badge } from "@/components/ui/badge";

export function ModeBadge({ mode }: { mode: Mode | string | null | undefined }) {
  if (!isMode(mode)) return null;

  const Icon = MODE_ICON[mode];
  const label = MODE_SHORT_LABEL[mode];

  return (
    <Badge variant="accent" className="shrink-0 px-2 py-0 text-[11px]" aria-label={MODE_ARIA_LABEL[mode]}>
      <Icon className="h-3 w-3" aria-hidden />
      {label}
    </Badge>
  );
}
