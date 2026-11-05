import type { Ref } from "react";
import { HardDrive, KeyRound, Cpu, Menu, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProviderStatus } from "@/lib/types";
import { StatusPill } from "./StatusPill";

export function TopBar({
  provider,
  compact = false,
  onOpenNavigation,
  menuButtonRef,
}: {
  provider?: ProviderStatus;
  compact?: boolean;
  onOpenNavigation?: () => void;
  menuButtonRef?: Ref<HTMLButtonElement>;
}) {
  const localModels = provider?.ollama_base_url ? 1 : 0;
  const externalStatus = provider?.external_search_service?.status ?? "unavailable";
  return (
    <header
      className={cn(
        "flex min-h-14 shrink-0 items-center justify-between gap-3 border-b border-border bg-bg py-2",
        compact ? "px-3" : "px-6",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        {compact ? (
          <>
            <button
              ref={menuButtonRef}
              type="button"
              aria-label="Open navigation"
              onClick={onOpenNavigation}
              className="grid h-9 w-9 shrink-0 place-items-center rounded-md text-text-muted hover:bg-bg-subtle focus-ring"
            >
              <Menu className="h-[18px] w-[18px]" aria-hidden />
            </button>
            <span className="truncate text-sm font-semibold text-text">Deep Research Studio</span>
          </>
        ) : (
          <span className="text-sm text-text-muted">Local-first AI research companion</span>
        )}
      </div>
      <div className={cn("flex min-w-0 items-center justify-end gap-2", compact && "flex-wrap")}>
        <StatusPill icon={HardDrive} label="Local-first" compactLabel={compact ? "Local" : undefined} connected />
        <StatusPill icon={KeyRound} label="BYOK" connected={Boolean(provider?.has_llm_key)} />
        <StatusPill
          icon={Search}
          label={`External search ${externalStatus}`}
          compactLabel={compact ? `Search ${externalStatus}` : undefined}
          connected={externalStatus === "ready"}
          className={externalStatus === "warn" ? "border-warning/50" : undefined}
          indicatorClassName={
            externalStatus === "warn"
              ? "bg-warning"
              : externalStatus === "unavailable"
                ? "bg-danger"
                : undefined
          }
        />
        <StatusPill
          icon={Cpu}
          label={`${localModels} local models connected`}
          compactLabel={compact ? `${localModels} models` : undefined}
          connected={localModels > 0}
        />
      </div>
    </header>
  );
}
