import { ArrowDown, ArrowUp, Plus, Trash2 } from "lucide-react";
import type { SearchProviderCapability, SearchProviderId, SearchUsageMode, SearchUsagePolicy } from "@/lib/types";
import { Button } from "@/components/ui/button";

const PROVIDERS: SearchProviderId[] = ["brave", "tavily", "exa", "firecrawl"];
const EXTRACT_PROVIDERS: Exclude<SearchProviderId, "brave">[] = ["tavily", "exa", "firecrawl"];

export const DEFAULT_SEARCH_USAGE_POLICY: SearchUsagePolicy = {
  usage_mode: "auto",
  fallback_chain: [...PROVIDERS],
};

export function SearchUsageEditor({
  value,
  onChange,
  allowGlobalDefault = false,
  disabled = false,
  compact = false,
  capabilities,
}: {
  value: SearchUsagePolicy | null;
  onChange: (value: SearchUsagePolicy | null) => void;
  allowGlobalDefault?: boolean;
  disabled?: boolean;
  compact?: boolean;
  capabilities?: Partial<Record<SearchProviderId, SearchProviderCapability>>;
}) {
  const policy = value ?? DEFAULT_SEARCH_USAGE_POLICY;
  const isEnabled = (provider: SearchProviderId) => capabilities?.[provider]?.enabled !== false;
  const isSelectable = (provider: SearchProviderId) =>
    isEnabled(provider) && (capabilities?.[provider]?.selectable ?? capabilities?.[provider]?.configured ?? true);
  const selectableSearchProviders = PROVIDERS.filter(isSelectable);
  const selectableExtractProviders = EXTRACT_PROVIDERS.filter(isSelectable);
  const visibleFallback = policy.fallback_chain.filter(isEnabled);
  const remaining = PROVIDERS.filter(
    (provider) => !policy.fallback_chain.includes(provider) && isSelectable(provider),
  );
  const canSelectSingle = selectableSearchProviders.length > 0;
  const canSelectDual = canSelectSingle && selectableExtractProviders.length > 0;

  const updateMode = (mode: SearchUsageMode) => {
    if (mode === "single") {
      if (!canSelectSingle) return;
      onChange({
        usage_mode: mode,
        primary_provider: policy.primary_provider && isSelectable(policy.primary_provider)
          ? policy.primary_provider
          : selectableSearchProviders[0],
        fallback_chain: policy.fallback_chain,
      });
    } else if (mode === "dual_role") {
      if (!canSelectDual) return;
      onChange({
        usage_mode: mode,
        search_provider: policy.search_provider && isSelectable(policy.search_provider)
          ? policy.search_provider
          : selectableSearchProviders[0],
        extract_provider: policy.extract_provider && isSelectable(policy.extract_provider)
          ? policy.extract_provider
          : selectableExtractProviders[0],
        fallback_chain: policy.fallback_chain,
      });
    } else {
      onChange({ usage_mode: mode, fallback_chain: policy.fallback_chain });
    }
  };

  const updateFallback = (next: SearchProviderId[]) => onChange({ ...policy, fallback_chain: next });

  return (
    <div className={compact ? "space-y-2" : "space-y-3"} data-testid="search-usage-editor">
      {allowGlobalDefault && (
        <div className="flex flex-wrap gap-2" role="group" aria-label="Search usage source">
          <Button
            type="button"
            size="sm"
            variant={value === null ? "default" : "outline"}
            disabled={disabled}
            onClick={() => onChange(null)}
          >
            Use global default
          </Button>
          <Button
            type="button"
            size="sm"
            variant={value !== null ? "default" : "outline"}
            disabled={disabled}
            onClick={() => onChange(value ?? { ...DEFAULT_SEARCH_USAGE_POLICY, fallback_chain: [...PROVIDERS] })}
          >
            Customize this session
          </Button>
        </div>
      )}

      {(!allowGlobalDefault || value !== null) && (
        <>
          <label className="block space-y-1 text-xs text-text-muted">
            <span>Search usage mode</span>
            <select
              aria-label="Search usage mode"
              value={policy.usage_mode}
              disabled={disabled}
              onChange={(event) => updateMode(event.target.value as SearchUsageMode)}
              className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
            >
              <option value="single" disabled={!canSelectSingle && policy.usage_mode !== "single"}>Single provider</option>
              <option value="dual_role" disabled={!canSelectDual && policy.usage_mode !== "dual_role"}>Dual role</option>
              <option value="auto">Auto (deterministic)</option>
            </select>
          </label>
          {(!canSelectSingle || !canSelectDual) && policy.usage_mode === "auto" && (
            <p className="text-xs text-text-faint">
              Enable an eligible External Search provider to use unavailable Single or Dual role modes.
            </p>
          )}

          {policy.usage_mode === "single" && (
            <ProviderSelect
              label="Primary provider"
              value={policy.primary_provider ?? "brave"}
              providers={PROVIDERS}
              disabled={disabled}
              capabilities={capabilities}
              onChange={(primary_provider) => onChange({ ...policy, primary_provider })}
            />
          )}
          {policy.usage_mode === "dual_role" && (
            <div className="grid gap-2 sm:grid-cols-2">
              <ProviderSelect
                label="Search provider"
                value={policy.search_provider ?? "brave"}
                providers={PROVIDERS}
                disabled={disabled}
                capabilities={capabilities}
                onChange={(search_provider) => onChange({ ...policy, search_provider })}
              />
              <ProviderSelect
                label="Extract provider"
                value={policy.extract_provider ?? "tavily"}
                providers={EXTRACT_PROVIDERS}
                disabled={disabled}
                capabilities={capabilities}
                onChange={(extract_provider) => onChange({ ...policy, extract_provider })}
              />
            </div>
          )}

          <div className="space-y-2">
            <p className="text-xs font-medium text-text">External fallback order</p>
            <ol className="space-y-1" aria-label="External fallback order">
              {visibleFallback.map((provider, index) => (
                <li key={provider} className="flex items-center gap-2 rounded border border-border bg-surface px-2 py-1.5 text-sm">
                  <span className="min-w-0 flex-1 text-text-body">
                    <span className="capitalize">{provider}</span>
                    {capabilities?.[provider] && (
                      <span className="ml-2 text-xs text-text-faint">
                        {capabilities[provider]?.configured ? "configured" : "not configured"}
                        {capabilities[provider]?.extract_ready ? " · search + extract" : " · search only"}
                        {capabilities[provider]?.selectable === false ? " · unavailable for new selection" : ""}
                      </span>
                    )}
                  </span>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    aria-label={`Move ${provider} up`}
                    disabled={disabled || index === 0}
                    onClick={() => updateFallback(moveVisible(policy.fallback_chain, visibleFallback, index, index - 1))}
                  >
                    <ArrowUp className="h-4 w-4" />
                  </Button>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    aria-label={`Move ${provider} down`}
                    disabled={disabled || index === visibleFallback.length - 1}
                    onClick={() => updateFallback(moveVisible(policy.fallback_chain, visibleFallback, index, index + 1))}
                  >
                    <ArrowDown className="h-4 w-4" />
                  </Button>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    aria-label={`Remove ${provider}`}
                    disabled={disabled}
                    onClick={() => updateFallback(policy.fallback_chain.filter((item) => item !== provider))}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </li>
              ))}
            </ol>
            {remaining.length > 0 && (
              <label className="flex items-center gap-2 text-xs text-text-muted">
                <span className="sr-only">Add fallback provider</span>
                <select
                  aria-label="Add fallback provider"
                  defaultValue=""
                  disabled={disabled}
                  onChange={(event) => {
                    const provider = event.target.value as SearchProviderId;
                    if (provider) updateFallback([...policy.fallback_chain, provider]);
                    event.target.value = "";
                  }}
                  className="h-9 min-w-0 flex-1 rounded-md border border-border bg-surface px-2 text-sm text-text"
                >
                  <option value="">Add provider…</option>
                  {remaining.map((provider) => (
                    <option key={provider} value={provider}>{provider}</option>
                  ))}
                </select>
                <Plus className="h-4 w-4" aria-hidden />
              </label>
            )}
            <p className="text-xs text-text-faint">
              Locked tail: cache → indexed_external. Provider order controls transport fallback, not evidence authority.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function ProviderSelect<T extends SearchProviderId>({
  label,
  value,
  providers,
  disabled,
  capabilities,
  onChange,
}: {
  label: string;
  value: T;
  providers: readonly T[];
  disabled: boolean;
  capabilities?: Partial<Record<SearchProviderId, SearchProviderCapability>>;
  onChange: (provider: T) => void;
}) {
  return (
    <label className="block space-y-1 text-xs text-text-muted">
      <span>{label}</span>
      <select
        aria-label={label}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value as T)}
        className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
      >
        {providers.filter((provider) => capabilities?.[provider]?.enabled !== false || provider === value).map((provider) => (
          <option
            key={provider}
            value={provider}
            disabled={
              capabilities?.[provider]?.enabled === false
              || (
                provider !== value
                && (capabilities?.[provider]?.selectable ?? capabilities?.[provider]?.configured) === false
              )
            }
          >
            {provider}
            {capabilities?.[provider]?.enabled === false
              ? " (disabled; runtime uses fallback)"
              : capabilities?.[provider]?.configured === false
                ? " (not configured)"
                : capabilities?.[provider]?.selectable === false
                  ? " (unavailable)"
                  : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

function moveVisible<T>(canonical: T[], visible: T[], from: number, to: number): T[] {
  if (to < 0 || to >= visible.length || from === to) return canonical;
  const fromIndex = canonical.indexOf(visible[from]);
  const toIndex = canonical.indexOf(visible[to]);
  if (fromIndex < 0 || toIndex < 0) return canonical;
  const next = [...canonical];
  [next[fromIndex], next[toIndex]] = [next[toIndex], next[fromIndex]];
  return next;
}
