import { Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";

export function ThemeToggle({ compact = false }: { compact?: boolean }) {
  const { theme, toggle } = useTheme();
  const next = theme === "light" ? "dark" : "light";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${next} theme`}
      title={compact ? `Theme: ${theme}` : undefined}
      className={cn(
        "inline-flex items-center rounded-md text-sm text-text-muted hover:bg-bg-subtle focus-ring",
        compact ? "grid h-8 w-8 place-items-center" : "gap-2 px-3 py-1.5",
      )}
    >
      {theme === "light" ? <Moon className="h-[18px] w-[18px]" aria-hidden /> : <Sun className="h-[18px] w-[18px]" aria-hidden />}
      {!compact && <span className="capitalize">{theme}</span>}
    </button>
  );
}
