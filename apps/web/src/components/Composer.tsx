import { useEffect, useMemo, useRef, useState } from "react";
import { SlidersHorizontal, Sparkles, ArrowUp, Loader2 } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type {
  CouncilStrategy,
  LibraryScope,
  LLMRegistryStatus,
  Mode,
  RunCostEstimate,
  RunCostEstimateRequest,
  SearchFreshness,
  SearchOptions,
  SearchUsagePolicy,
  ThinkingEffort,
} from "@/lib/types";
import { chatRefs } from "@/lib/registry";
import { effortLabel, THINKING_EFFORTS } from "@/lib/thinking";
import { CONTEXT_PRESETS } from "@/lib/contextBudget";
import { cn } from "@/lib/utils";
import { BudgetPreviewBar } from "./BudgetPreviewBar";
import { Textarea, Input } from "./ui/input";
import { Button } from "./ui/button";
import { SearchUsageEditor } from "./SearchUsageEditor";
import { useProvider } from "./AppShell";
import { IS_DEMO } from "@/lib/runtime";

export const COUNCIL_SIZE_MIN = 2;
export const COUNCIL_SIZE_MAX = 5;
export const COUNCIL_SIZE_DEFAULT = 3;

const COUNCIL_STRATEGY_OPTIONS: { value: CouncilStrategy; title: string; description: string }[] = [
  {
    value: "parallel_vote",
    title: "Parallel vote",
    description: "Several models answer independently; fastest path, no chair synthesis.",
  },
  {
    value: "judge_synthesis",
    title: "Judge synthesis",
    description: "Chair merges drafts into structured consensus (default).",
  },
  {
    value: "debate_once",
    title: "One critique round",
    description: "Members critique peers once, then chair synthesizes.",
  },
];

const FRESHNESS_OPTIONS: { value: "" | SearchFreshness; label: string }[] = [
  { value: "", label: "Any time" },
  { value: "pd", label: "Past day" },
  { value: "pw", label: "Past week" },
  { value: "pm", label: "Past month" },
  { value: "py", label: "Past year" },
];

export interface ComposerOptions {
  budget?: Record<string, number>;
  llm_model?: string;
  council_models?: string[];
  council_size?: number;
  council_strategy?: CouncilStrategy;
  temperature?: number;
  thinking_effort?: ThinkingEffort;
  memory_policy?: "off" | "workspace" | "workspace_user";
  library_scope?: LibraryScope;
  context_budget?: { preset: string };
  search_options?: SearchOptions;
  search_usage_policy?: SearchUsagePolicy | null;
}

/** Large query composer card with toolbar, prompt enhancement and budget/model options. */
export function Composer({
  onSubmit,
  busy,
  mode,
  placeholder = "Ask a research question…",
  onQueryChange,
  defaultUseMemory = false,
  initialSearchUsagePolicy,
  onSearchUsagePolicyChange,
}: {
  onSubmit: (query: string, options?: ComposerOptions) => void;
  busy?: boolean;
  mode?: Mode;
  placeholder?: string;
  /** Fires on every keystroke so callers can drive live features (e.g. semantic recall). */
  onQueryChange?: (query: string) => void;
  /** Initial state of the "Use workspace memory" toggle; callers (follow-ups) may default on. */
  defaultUseMemory?: boolean;
  /** Session override loaded by the thread; undefined/null both initially follow global. */
  initialSearchUsagePolicy?: SearchUsagePolicy | null;
  /** Session threads persist changes before allowing a run to start. */
  onSearchUsagePolicyChange?: (policy: SearchUsagePolicy | null) => Promise<void>;
}) {
  const [value, setValue] = useState("");
  const [enhancing, setEnhancing] = useState(false);
  const [error, setError] = useState<string>();
  const [showOptions, setShowOptions] = useState(false);
  const [maxSearches, setMaxSearches] = useState("");
  const [topK, setTopK] = useState("");
  const [registry, setRegistry] = useState<LLMRegistryStatus>();
  const [model, setModel] = useState("");
  const [council, setCouncil] = useState<string[]>([]);
  const [councilSize, setCouncilSize] = useState(COUNCIL_SIZE_DEFAULT);
  const [councilStrategy, setCouncilStrategy] = useState<CouncilStrategy>("judge_synthesis");
  const [temperature, setTemperature] = useState("");
  // "" = use the model preset; an effort enum overrides it for this run.
  const [thinkingDepth, setThinkingDepth] = useState<"" | ThinkingEffort>("");
  const [useMemory, setUseMemory] = useState(defaultUseMemory);
  const [libraryScope, setLibraryScope] = useState<LibraryScope>("off");
  // "" = use the global Settings budget; a preset id overrides it for this run only.
  const [contextPreset, setContextPreset] = useState("");
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false);
  const [includeDomains, setIncludeDomains] = useState("");
  const [excludeDomains, setExcludeDomains] = useState("");
  const [freshness, setFreshness] = useState<"" | SearchFreshness>("");
  const [extraSubqueries, setExtraSubqueries] = useState("");
  const [country, setCountry] = useState("");
  const [searchLang, setSearchLang] = useState("");
  const [uiLanguage, setUiLanguage] = useState("");
  const [costEstimate, setCostEstimate] = useState<RunCostEstimate | null>(null);
  const [searchUsagePolicy, setSearchUsagePolicy] = useState<SearchUsagePolicy | null>(
    initialSearchUsagePolicy ?? null,
  );
  const [globalSearchUsagePolicy, setGlobalSearchUsagePolicy] = useState<SearchUsagePolicy>();
  const [searchUsageTouched, setSearchUsageTouched] = useState(false);
  const [searchUsageSaving, setSearchUsageSaving] = useState(false);
  const [searchUsageError, setSearchUsageError] = useState<string>();
  const searchUsageSaveTail = useRef<Promise<void>>(Promise.resolve());
  const searchUsageSaveSequence = useRef(0);
  const searchUsageSavePending = useRef(false);
  const providerStatus = useProvider();

  useEffect(() => {
    api.llmRegistry().then(setRegistry).catch(() => undefined);
    api.getSearchUsage().then(setGlobalSearchUsagePolicy).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (searchUsageSavePending.current) return;
    setSearchUsagePolicy(initialSearchUsagePolicy ?? null);
    setSearchUsageTouched(false);
    setSearchUsageError(undefined);
  }, [initialSearchUsagePolicy]);

  const updateSearchUsagePolicy = (policy: SearchUsagePolicy | null) => {
    setSearchUsagePolicy(policy);
    setSearchUsageTouched(true);
    setSearchUsageError(undefined);
    if (!onSearchUsagePolicyChange) return;
    const sequence = searchUsageSaveSequence.current + 1;
    searchUsageSaveSequence.current = sequence;
    searchUsageSavePending.current = true;
    setSearchUsageSaving(true);
    const save = searchUsageSaveTail.current
      .catch(() => undefined)
      .then(() => onSearchUsagePolicyChange(policy));
    searchUsageSaveTail.current = save;
    void save.then(
      () => {
        if (searchUsageSaveSequence.current !== sequence) return;
        searchUsageSavePending.current = false;
        setSearchUsageSaving(false);
      },
      (cause) => {
        if (searchUsageSaveSequence.current !== sequence) return;
        searchUsageSavePending.current = false;
        setSearchUsageError(cause instanceof ApiError ? cause.message : "Search strategy was not saved.");
        setSearchUsageSaving(false);
      },
    );
  };

  // Only chat/reasoning models are selectable; embedders are excluded.
  const refs = useMemo(() => chatRefs(registry), [registry]);
  const externalPathUnavailable = useMemo(() => {
    const effectivePolicy = searchUsagePolicy ?? globalSearchUsagePolicy;
    if (!effectivePolicy) {
      return providerStatus?.external_search_service?.status === "unavailable";
    }
    const candidates = new Set(effectivePolicy.fallback_chain);
    if (effectivePolicy.primary_provider) candidates.add(effectivePolicy.primary_provider);
    if (effectivePolicy.search_provider) candidates.add(effectivePolicy.search_provider);
    return ![...candidates].some((provider) => providerStatus?.search_providers?.[provider]?.runtime_eligible);
  }, [globalSearchUsagePolicy, providerStatus, searchUsagePolicy]);

  const toggleCouncil = (ref: string) =>
    setCouncil((prev) => (prev.includes(ref) ? prev.filter((r) => r !== ref) : [...prev, ref]));

  const buildOptions = (): ComposerOptions | undefined => {
    const options: ComposerOptions = {};
    const budget: Record<string, number> = {};
    if (maxSearches.trim()) budget.max_searches = Number(maxSearches);
    if (topK.trim()) budget.top_k = Number(topK);
    if (Object.keys(budget).length) options.budget = budget;
    if (model) options.llm_model = model;
    if (temperature.trim()) options.temperature = Number(temperature);
    if (thinkingDepth) options.thinking_effort = thinkingDepth;
    // Explicit policy avoids backend/frontend double-default drift on RunRequest replay.
    if (useMemory) options.memory_policy = "workspace";
    if (libraryScope !== "off") options.library_scope = libraryScope;
    // Empty preset keeps the global Settings budget; a chosen preset is a per-run override.
    if (contextPreset) options.context_budget = { preset: contextPreset };
    const searchOptions = buildSearchOptions({
      includeDomains,
      excludeDomains,
      freshness,
      extraSubqueries,
      country,
      searchLang,
      uiLanguage,
    });
    if (searchOptions) options.search_options = searchOptions;
    if (searchUsageTouched) options.search_usage_policy = searchUsagePolicy;
    if (mode === "council") {
      options.council_size = councilSize;
      if (council.length) options.council_models = council;
      options.council_strategy = councilStrategy;
    }
    return Object.keys(options).length ? options : undefined;
  };

  const submit = () => {
    const q = value.trim();
    if (q && !busy && !searchUsageSaving && !searchUsageError) onSubmit(q, buildOptions());
  };

  const previewOverrides = useMemo(
    () => ({
      maxSearches: maxSearches.trim() ? Number(maxSearches) : undefined,
      topK: topK.trim() ? Number(topK) : undefined,
      councilSize: mode === "council" ? councilSize : undefined,
      councilStrategy: mode === "council" ? councilStrategy : undefined,
    }),
    [maxSearches, topK, mode, councilSize, councilStrategy],
  );

  const estimatePayload = useMemo<RunCostEstimateRequest | null>(() => {
    if (!mode) return null;
    const payload: RunCostEstimateRequest = { mode };
    const budget: Record<string, number> = {};
    if (maxSearches.trim()) budget.max_searches = Number(maxSearches);
    if (topK.trim()) budget.top_k = Number(topK);
    if (Object.keys(budget).length) payload.budget = budget;
    if (model) payload.llm_model = model;
    const searchOptions = buildSearchOptions({
      includeDomains,
      excludeDomains,
      freshness,
      extraSubqueries,
      country,
      searchLang,
      uiLanguage,
    });
    if (searchOptions) payload.search_options = searchOptions;
    if (mode === "council") {
      payload.council_size = councilSize;
      payload.council_strategy = councilStrategy;
      if (council.length) payload.council_models = council;
    }
    return payload;
  }, [
    mode,
    maxSearches,
    topK,
    model,
    includeDomains,
    excludeDomains,
    freshness,
    extraSubqueries,
    country,
    searchLang,
    uiLanguage,
    councilSize,
    councilStrategy,
    council,
  ]);

  useEffect(() => {
    if (!estimatePayload) {
      setCostEstimate(null);
      return;
    }
    let active = true;
    api
      .estimateRunCost(estimatePayload)
      .then((estimate) => {
        if (active) setCostEstimate(estimate);
      })
      .catch(() => {
        if (active) setCostEstimate(null);
      });
    return () => {
      active = false;
    };
  }, [estimatePayload]);

  const enhance = async () => {
    const q = value.trim();
    if (!q || enhancing) return;
    setEnhancing(true);
    setError(undefined);
    try {
      const { enhanced } = await api.enhancePrompt(q);
      setValue(enhanced);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setEnhancing(false);
    }
  };

  return (
    <div data-anim className="rounded-md border border-border bg-surface p-3 shadow-card">
      {IS_DEMO && (
        <div className="mb-3 flex flex-wrap gap-2" aria-label="Preset preview questions">
          {[
            "Which search query operators does SQLite FTS5 support? Cite the documentation.",
            "Compare SQLite FTS5 prefix searches, phrase searches and NEAR queries.",
          ].map((preset) => (
            <Button key={preset} type="button" size="sm" variant="outline" onClick={() => { setValue(preset); onQueryChange?.(preset); }}>
              {preset}
            </Button>
          ))}
        </div>
      )}
      <Textarea
        rows={3}
        value={value}
        placeholder={placeholder}
        readOnly={IS_DEMO}
        onChange={(e) => {
          setValue(e.target.value);
          onQueryChange?.(e.target.value);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
        }}
        className="border-0 focus-visible:ring-0"
        aria-label="Research question"
      />
      <div className="flex items-center justify-between px-1 pt-1">
        <div className="flex items-center gap-1 text-text-muted">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Options"
            aria-pressed={showOptions}
            type="button"
            onClick={() => setShowOptions((s) => !s)}
          >
            <SlidersHorizontal className="h-[18px] w-[18px]" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Enhance prompt"
            type="button"
            disabled={IS_DEMO || enhancing || !value.trim()}
            onClick={enhance}
          >
            {enhancing ? <Loader2 className="h-[18px] w-[18px] animate-spin" /> : <Sparkles className="h-[18px] w-[18px]" />}
          </Button>
        </div>
        <Button
          size="icon"
          aria-label="Send"
          disabled={busy || searchUsageSaving || Boolean(searchUsageError) || !value.trim()}
          onClick={submit}
          type="button"
        >
          <ArrowUp className="h-[18px] w-[18px]" />
        </Button>
      </div>

      {mode && (
        <BudgetPreviewBar
          mode={mode}
          overrides={previewOverrides}
          costEstimate={costEstimate}
        />
      )}
      <p className="mt-2 px-1 text-xs text-text-muted" aria-label="Search strategy summary">
        Search strategy: {searchUsagePolicy === null
          ? "global default"
          : `${searchUsagePolicy.usage_mode} · session override`}
      </p>
      {externalPathUnavailable && (
        <p className="mt-1 px-1 text-xs text-warning" role="status">
          External search is unavailable for this strategy. You can still run with local, Memory, cache, or indexed_external evidence; grounding decides whether an answer can complete.
        </p>
      )}

      {showOptions && (
        <div className="mt-3 grid grid-cols-2 gap-3 rounded-md border border-border bg-bg-subtle p-3">
          <label className="space-y-1 text-xs text-text-muted">
            <span>Max searches</span>
            <Input
              type="number"
              min={1}
              value={maxSearches}
              onChange={(e) => setMaxSearches(e.target.value)}
              placeholder="preset"
              aria-label="Max searches"
            />
          </label>
          <label className="space-y-1 text-xs text-text-muted">
            <span>Top-K passages</span>
            <Input
              type="number"
              min={1}
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              placeholder="preset"
              aria-label="Top-K passages"
            />
          </label>
          <label className="col-span-2 space-y-1 text-xs text-text-muted">
            <span>Model</span>
            <select
              aria-label="Model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
            >
              <option value="">Default ({registry?.roles.default || "configured"})</option>
              {refs.map((r) => (
                <option key={r.ref} value={r.ref}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs text-text-muted">
            <span>Temperature</span>
            <Input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
              placeholder="preset"
              aria-label="Temperature"
            />
          </label>
          <label className="space-y-1 text-xs text-text-muted">
            <span>Thinking depth</span>
            <select
              aria-label="Thinking depth"
              value={thinkingDepth}
              onChange={(e) => setThinkingDepth(e.target.value as "" | ThinkingEffort)}
              className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
            >
              <option value="">Default (preset)</option>
              {THINKING_EFFORTS.map((eff) => (
                <option key={eff} value={eff}>
                  {effortLabel(eff)}
                </option>
              ))}
            </select>
          </label>
          <label className="col-span-2 flex items-center gap-2 text-xs text-text">
            <input
              type="checkbox"
              aria-label="Use workspace memory"
              checked={useMemory}
              onChange={(e) => setUseMemory(e.target.checked)}
            />
            <span>Use workspace memory (inject saved preferences/background as context)</span>
          </label>
          <label className="col-span-2 space-y-1 text-xs text-text-muted">
            <span>Library context</span>
            <select
              aria-label="Library context"
              value={libraryScope}
              onChange={(e) => setLibraryScope(e.target.value as LibraryScope)}
              className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
            >
              <option value="off">Off</option>
              <option value="notes">Notes</option>
              <option value="reports">Reports</option>
              <option value="all">Notes + reports</option>
            </select>
          </label>
          <label className="col-span-2 space-y-1 text-xs text-text-muted">
            <span>Context budget</span>
            <select
              aria-label="Context budget"
              value={contextPreset}
              onChange={(e) => setContextPreset(e.target.value)}
              className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
            >
              <option value="">Use global setting</option>
              {CONTEXT_PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <div className="col-span-2 rounded-md border border-border bg-surface p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-sm font-medium text-text">Advanced search</p>
                <p className="text-xs text-text-faint">Optional domain, freshness, and multi-search query controls.</p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                type="button"
                aria-expanded={showAdvancedSearch}
                aria-controls="advanced-search-options"
                onClick={() => setShowAdvancedSearch((value) => !value)}
              >
                {showAdvancedSearch ? "Hide" : "Show"}
              </Button>
            </div>
            {showAdvancedSearch && (
              <div id="advanced-search-options" className="mt-3 grid gap-3 sm:grid-cols-2">
                <div className="sm:col-span-2 rounded-md border border-border bg-bg-subtle p-3">
                  <div className="mb-2">
                    <p className="text-sm font-medium text-text">Search provider strategy</p>
                    <p className="text-xs text-text-faint">Changes here apply only to this session.</p>
                  </div>
                  <SearchUsageEditor
                    value={searchUsagePolicy}
                    allowGlobalDefault
                    compact
                    disabled={IS_DEMO || searchUsageSaving}
                    capabilities={providerStatus?.search_providers}
                    onChange={(policy) => void updateSearchUsagePolicy(policy)}
                  />
                  {searchUsageSaving && <p role="status" className="mt-2 text-xs text-text-muted">Saving session strategy…</p>}
                  {searchUsageError && <p role="alert" className="mt-2 text-xs text-danger">{searchUsageError}</p>}
                </div>
                <label className="space-y-1 text-xs text-text-muted">
                  <span>Include domains</span>
                  <Input
                    value={includeDomains}
                    onChange={(e) => setIncludeDomains(e.target.value)}
                    placeholder="example.com, docs.example.com"
                    aria-label="Include domains"
                  />
                </label>
                <label className="space-y-1 text-xs text-text-muted">
                  <span>Exclude domains</span>
                  <Input
                    value={excludeDomains}
                    onChange={(e) => setExcludeDomains(e.target.value)}
                    placeholder="spam.example"
                    aria-label="Exclude domains"
                  />
                </label>
                <label className="space-y-1 text-xs text-text-muted">
                  <span>Freshness</span>
                  <select
                    aria-label="Search freshness"
                    value={freshness}
                    onChange={(e) => setFreshness(e.target.value as "" | SearchFreshness)}
                    className="h-9 w-full rounded-md border border-border bg-surface px-2 text-sm text-text"
                  >
                    {FRESHNESS_OPTIONS.map((option) => (
                      <option key={option.value || "any"} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1 text-xs text-text-muted">
                  <span>Search country</span>
                  <Input
                    value={country}
                    onChange={(e) => setCountry(e.target.value)}
                    placeholder="US"
                    maxLength={2}
                    aria-label="Search country"
                  />
                </label>
                <label className="space-y-1 text-xs text-text-muted">
                  <span>Search language</span>
                  <Input
                    value={searchLang}
                    onChange={(e) => setSearchLang(e.target.value)}
                    placeholder="zh-hans"
                    maxLength={35}
                    aria-label="Search language"
                  />
                </label>
                <label className="space-y-1 text-xs text-text-muted sm:col-span-2">
                  <span>搜索结果界面语言</span>
                  <Input
                    value={uiLanguage}
                    onChange={(e) => setUiLanguage(e.target.value)}
                    placeholder="zh-Hans-CN"
                    maxLength={35}
                    aria-label="搜索结果界面语言"
                  />
                  <span className="block text-text-faint">仅影响搜索提供方的结果界面语言，不控制应用界面或报告输出语言。</span>
                </label>
                <label className="space-y-1 text-xs text-text-muted sm:col-span-2">
                  <span>Extra subqueries</span>
                  <textarea
                    value={extraSubqueries}
                    onChange={(e) => setExtraSubqueries(e.target.value)}
                    placeholder="One extra multi-search query per line"
                    aria-label="Extra subqueries"
                    rows={3}
                    className="min-h-20 w-full resize-y rounded-md border border-border bg-surface px-3 py-2 text-sm text-text outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  />
                </label>
              </div>
            )}
          </div>
          {mode === "council" && (
            <div className="col-span-2 space-y-2 text-xs text-text-muted">
              <label className="space-y-1">
                <span>Council members ({COUNCIL_SIZE_MIN}–{COUNCIL_SIZE_MAX})</span>
                <Input
                  type="number"
                  min={COUNCIL_SIZE_MIN}
                  max={COUNCIL_SIZE_MAX}
                  value={councilSize}
                  onChange={(e) =>
                    setCouncilSize(
                      Math.min(COUNCIL_SIZE_MAX, Math.max(COUNCIL_SIZE_MIN, Number(e.target.value) || COUNCIL_SIZE_DEFAULT)),
                    )
                  }
                  aria-label="Council member count"
                />
              </label>
              {refs.length > 0 && (
                <div className="space-y-1">
                  <span>Candidate models (optional)</span>
                  <div className="flex flex-wrap gap-2">
                    {refs.map((r) => (
                      <label key={r.ref} className="flex items-center gap-1 rounded border border-border px-2 py-1 text-text">
                        <input
                          type="checkbox"
                          aria-label={`council member ${r.ref}`}
                          checked={council.includes(r.ref)}
                          onChange={() => toggleCouncil(r.ref)}
                        />
                        {r.label}
                      </label>
                    ))}
                  </div>
                </div>
              )}
              <div className="space-y-1">
                <span>Council strategy</span>
                <div className="grid gap-2 sm:grid-cols-3">
                  {COUNCIL_STRATEGY_OPTIONS.map((option) => {
                    const active = councilStrategy === option.value;
                    return (
                      <button
                        key={option.value}
                        type="button"
                        aria-label={`Council strategy ${option.title}`}
                        aria-pressed={active}
                        data-selected={active ? "true" : undefined}
                        onClick={() => setCouncilStrategy(option.value)}
                        className={cn(
                          "rounded-md border p-2 text-left shadow-card transition-colors focus-ring",
                          active
                            ? "border-accent bg-accent-soft ring-2 ring-accent/25"
                            : "border-border bg-surface text-text-muted hover:border-border-strong",
                        )}
                      >
                        <span className={cn("block font-semibold", active ? "text-accent" : "text-text")}>
                          {option.title}
                        </span>
                        <span className={cn("mt-0.5 block leading-snug", active ? "text-text-body" : "text-text-faint")}>
                          {option.description}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
              <p className="text-text-faint">
                DeepResearch assigns each member a model and a distinct perspective for this question.
              </p>
            </div>
          )}
        </div>
      )}

      {error && (
        <p role="alert" className="mt-2 px-1 text-sm text-danger">
          {error}
        </p>
      )}
    </div>
  );
}

function splitCsv(value: string): string[] {
  return value
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function splitLines(value: string): string[] {
  return value
    .split(/\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildSearchOptions({
  includeDomains,
  excludeDomains,
  freshness,
  extraSubqueries,
  country,
  searchLang,
  uiLanguage,
}: {
  includeDomains: string;
  excludeDomains: string;
  freshness: "" | SearchFreshness;
  extraSubqueries: string;
  country: string;
  searchLang: string;
  uiLanguage: string;
}): SearchOptions | undefined {
  const options: SearchOptions = {};
  const include = splitCsv(includeDomains);
  const exclude = splitCsv(excludeDomains);
  const extra = splitLines(extraSubqueries);
  if (include.length) options.include_domains = include;
  if (exclude.length) options.exclude_domains = exclude;
  if (freshness) options.freshness = freshness;
  if (extra.length) options.extra_subqueries = extra;
  if (country.trim()) options.country = country.trim();
  if (searchLang.trim()) options.search_lang = searchLang.trim();
  if (uiLanguage.trim()) options.ui_language = uiLanguage.trim();
  return Object.keys(options).length ? options : undefined;
}
