import type { LucideIcon } from "lucide-react";
import type { Mode } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface ModeOption {
  mode: Mode;
  title: string;
  subtitle: string;
  icon: LucideIcon;
}

export function ModeCard({
  option,
  active,
  onSelect,
}: {
  option: ModeOption;
  active: boolean;
  onSelect: (mode: Mode) => void;
}) {
  const { mode, title, subtitle, icon: Icon } = option;
  return (
    <button
      type="button"
      data-anim
      aria-pressed={active}
      onClick={() => onSelect(mode)}
      className={cn(
        "flex flex-col items-start gap-2 rounded-md border bg-surface p-4 text-left shadow-card transition-colors focus-ring",
        active ? "border-accent ring-1 ring-[color:var(--ring)]" : "border-border hover:border-border-strong",
      )}
    >
      <Icon className={cn("h-5 w-5", active ? "text-accent" : "text-text-muted")} aria-hidden />
      <span className="text-sm font-semibold text-text">{title}</span>
      <span className="text-xs leading-snug text-text-muted">{subtitle}</span>
    </button>
  );
}
