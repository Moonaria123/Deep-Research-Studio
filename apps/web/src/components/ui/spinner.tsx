import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

/** Accessible loading spinner with a screen-reader label. */
export function Spinner({ className, label = "Loading" }: { className?: string; label?: string }) {
  return (
    <span role="status" className="inline-flex items-center gap-2 text-text-muted">
      <Loader2 className={cn("h-4 w-4 animate-spin", className)} aria-hidden />
      <span className="sr-only">{label}</span>
    </span>
  );
}
