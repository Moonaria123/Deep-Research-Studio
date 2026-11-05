import { Database, Search, Globe, Cpu, Brain } from "lucide-react";
import type { ProviderStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

function Chip({ icon: Icon, label, on }: { icon: typeof Database; label: string; on: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs text-text-muted">
      <Icon className="h-3.5 w-3.5" aria-hidden />
      {label}
      <span className={cn("h-1.5 w-1.5 rounded-full", on ? "bg-success" : "bg-text-faint")} aria-hidden />
      <span className="sr-only">{on ? "connected" : "not connected"}</span>
    </span>
  );
}

export function ConnectorStrip({ provider }: { provider?: ProviderStatus }) {
  return (
    <div data-anim className="flex flex-wrap gap-2">
      <Chip icon={Database} label="Local cache" on />
      <Chip icon={Search} label="Brave" on={Boolean(provider?.search_providers?.brave.runtime_eligible)} />
      <Chip icon={Search} label="Tavily" on={Boolean(provider?.search_providers?.tavily.runtime_eligible)} />
      <Chip icon={Search} label="Exa" on={Boolean(provider?.search_providers?.exa.runtime_eligible)} />
      <Chip icon={Globe} label="Firecrawl" on={Boolean(provider?.search_providers?.firecrawl.runtime_eligible)} />
      <Chip icon={Cpu} label="Ollama" on={Boolean(provider?.ollama_base_url)} />
      <Chip icon={Brain} label="Cross-session memory" on />
    </div>
  );
}
