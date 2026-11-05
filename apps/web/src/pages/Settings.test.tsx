import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import { Settings } from "./Settings";
import { api } from "@/lib/api";
import type { ProviderStatus } from "@/lib/types";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  const budgetConfig = {
    preset: "balanced",
    max_chars: 6000,
    turns: 6,
    recent_chars: 3500,
    semantic_chars: 1500,
    memory_chars: 1000,
    recent_share: 58,
    semantic_share: 25,
    memory_share: 17,
  };
  const budgetState = { effective: budgetConfig, saved: null, env: null, default: budgetConfig, source: "default" };
  return {
    ...actual,
    api: {
      providers: vi.fn(),
      updateProviders: vi.fn(),
      saveSearchProviders: vi.fn(),
      toggleSearchProvider: vi.fn(),
      health: vi.fn().mockResolvedValue({ status: "ok", fts5: true, job_backend: "thread", redis_reachable: null }),
      llmRegistry: vi.fn().mockResolvedValue({ providers: [], roles: { default: "", embed: "", memory: "", council: [] } }),
      testTools: vi.fn(),
      getSearchUsage: vi.fn().mockResolvedValue({
        usage_mode: "auto",
        fallback_chain: ["brave", "tavily", "exa", "firecrawl"],
      }),
      updateSearchUsage: vi.fn(),
      getDeepCheckpointSettings: vi.fn().mockResolvedValue({
        auto_checkpoint_enabled: false,
      }),
      updateDeepCheckpointSettings: vi.fn().mockResolvedValue({
        auto_checkpoint_enabled: false,
      }),
      getContextBudget: vi.fn().mockResolvedValue(budgetState),
      updateContextBudget: vi.fn().mockResolvedValue(budgetState),
      recommendContextBudget: vi.fn(),
      getRetrievalSettings: vi.fn().mockResolvedValue({
        saved: { chroma_persist_enabled: true },
        effective: {
          chroma_persist_enabled: true,
          chroma_persist_mode: "disk",
          chroma_persist_path: ".../data/chroma",
          ephemeral_forced: false,
        },
        restart_required: false,
      }),
      updateRetrievalSettings: vi.fn().mockResolvedValue({
        saved: { chroma_persist_enabled: false },
        effective: {
          chroma_persist_enabled: false,
          chroma_persist_mode: "ephemeral",
          chroma_persist_path: null,
          ephemeral_forced: false,
        },
        restart_required: true,
      }),
      getPostRunNotes: vi.fn().mockResolvedValue({ enabled_modes: [] }),
      updatePostRunNotes: vi.fn().mockResolvedValue({ enabled_modes: [] }),
      getMemorySettings: vi.fn().mockResolvedValue({
        post_turn_memory_refresh_enabled: true,
        workspace_memory_refresh_strategy: "incremental",
        memory_thinking_enabled: false,
        format_probe: null,
      }),
      updateMemorySettings: vi.fn(),
      probeMemoryFormat: vi.fn(),
      getKnowledgeGraphSettings: vi.fn().mockResolvedValue({
        post_run_rebuild_enabled: true,
        wiki_auto_refresh_after_kg_rebuild: false,
      }),
      updateKnowledgeGraphSettings: vi.fn().mockResolvedValue({
        post_run_rebuild_enabled: true,
        wiki_auto_refresh_after_kg_rebuild: false,
      }),
      purgeCache: vi.fn().mockResolvedValue({ purged_count: 0, scope: "all" }),
    },
  };
});

const unconfigured: ProviderStatus = {
  llm_base_url: "",
  llm_model: "",
  has_llm_key: false,
  brave_ready: false,
  firecrawl_ready: false,
  embed_model: "",
  ollama_base_url: "",
};

const configured: ProviderStatus = {
  ...unconfigured,
  brave_ready: true,
  firecrawl_ready: true,
  provider_config_revision: 3,
  search_providers: {
    brave: { configured: true, search_ready: true, extract_ready: false, enabled: true, runtime_eligible: true, selectable: true, credential_source: "config", probe_state: "available", lamp: "green" },
    firecrawl: { configured: true, search_ready: true, extract_ready: true, enabled: true, runtime_eligible: true, selectable: true, credential_source: "config", probe_state: "available", lamp: "green" },
    tavily: { configured: false, search_ready: false, extract_ready: false, enabled: true, runtime_eligible: false, selectable: false, credential_source: "none", probe_state: "not_configured", lamp: "gray" },
    exa: { configured: false, search_ready: false, extract_ready: false, enabled: true, runtime_eligible: false, selectable: false, credential_source: "none", probe_state: "not_configured", lamp: "gray" },
  },
  external_search_service: { status: "ready", runtime_ready: true },
};

describe("Settings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("scopes the only search-provider save to its card and has no page-wide save", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    render(<Settings />);
    await waitFor(() => expect(api.providers).toHaveBeenCalled());

    const form = screen.getByRole("form", { name: /search providers settings/i });
    expect(within(form).getByRole("button", { name: /save search providers/i })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /save search providers/i })).toHaveLength(1);
    expect(screen.queryByRole("button", { name: /save settings/i })).not.toBeInTheDocument();
    expect(form).not.toContainElement(screen.getByRole("button", { name: /save search usage/i }));

    expect(within(form).getByRole("button", { name: /test brave/i })).toHaveAttribute("type", "button");
    expect(within(form).getByRole("button", { name: /disable brave/i })).toHaveAttribute("type", "button");
    expect(within(form).getByRole("button", { name: /remove brave credential/i })).toHaveAttribute("type", "button");

    fireEvent.change(screen.getByLabelText("Brave Search API key"), { target: { value: "BSA-form-boundary" } });
    fireEvent.submit(form);
    await waitFor(() => expect(api.saveSearchProviders).toHaveBeenCalledTimes(1));
  });

  it("keeps automatic Deep checkpoints off by default and saves the explicit preference", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.getDeepCheckpointSettings).mockResolvedValue({
      auto_checkpoint_enabled: false,
    });
    vi.mocked(api.updateDeepCheckpointSettings).mockResolvedValue({
      auto_checkpoint_enabled: true,
    });

    render(<Settings />);

    const toggle = await screen.findByRole("checkbox", {
      name: /automatically save at safe deep research boundaries/i,
    });
    expect(toggle).not.toBeChecked();
    expect(screen.getByText(/unexpected interruption recovery is not available/i)).toBeInTheDocument();

    fireEvent.click(toggle);
    fireEvent.click(screen.getByRole("button", { name: /save deep checkpoint setting/i }));

    await waitFor(() =>
      expect(api.updateDeepCheckpointSettings).toHaveBeenCalledWith({
        auto_checkpoint_enabled: true,
      }),
    );
    expect(await screen.findByText(/preference saved for new runs/i)).toBeInTheDocument();
  });

  it("no longer renders the legacy Language model (BYOK) card", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    render(<Settings />);
    await waitFor(() => expect(api.providers).toHaveBeenCalled());
    expect(screen.queryByText(/Language model \(BYOK\)/i)).not.toBeInTheDocument();
    // LLM configuration now lives only in the provider registry card.
    expect(screen.getAllByText(/LLM providers/i).length).toBeGreaterThan(0);
  });

  it("shows the read-only job backend status from /health", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.health).mockResolvedValue({
      status: "ok",
      fts5: true,
      db_encrypted: true,
      auth_mode: "token",
      vector_backend: "chroma",
      job_backend: "rq",
      redis_reachable: true,
    });
    render(<Settings />);
    await waitFor(() => expect(api.health).toHaveBeenCalled());
    expect(await screen.findByText(/Queue \(RQ\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Redis connected/i)).toBeInTheDocument();
    // Database-at-rest status surfaces SQLCipher encryption (DR-ITER-023).
    expect(screen.getByText(/Encrypted \(SQLCipher\)/i)).toBeInTheDocument();
    // API auth mode surfaces token enforcement (DR-ITER-027).
    expect(screen.getByText(/Token required/i)).toBeInTheDocument();
    // Vector backend surfaces the active ANN index (DR-ITER-020).
    expect(screen.getByText(/ANN \(Chroma\)/i)).toBeInTheDocument();
    // No switch control: the backend is read-only.
    expect(screen.queryByRole("combobox", { name: /job backend/i })).not.toBeInTheDocument();
  });

  it("loads and saves Chroma retrieval persistence settings", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.health).mockResolvedValue({
      status: "ok",
      fts5: true,
      vector_backend: "chroma",
      chroma_persist_enabled: true,
      chroma_persist_mode: "disk",
      chroma_persist_path: ".../data/chroma",
      job_backend: "thread",
      redis_reachable: null,
    });
    render(<Settings />);
    await waitFor(() => expect(api.getRetrievalSettings).toHaveBeenCalled());

    const toggle = await screen.findByRole("switch", { name: /persist chroma vector index on disk/i });
    expect(toggle).toHaveAttribute("aria-checked", "true");
    expect(toggle).not.toBeDisabled();
    fireEvent.click(toggle);
    fireEvent.click(screen.getByRole("button", { name: /save retrieval/i }));

    await waitFor(() =>
      expect(api.updateRetrievalSettings).toHaveBeenCalledWith({
        chroma_persist_enabled: false,
        kg_aware_retrieval_enabled: false,
      }),
    );
    expect(await screen.findByText(/^Saved\. Restart the API/i)).toBeInTheDocument();
  });

  it("saves KG candidate discovery independently without requiring an API restart", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.updateRetrievalSettings).mockResolvedValueOnce({
      saved: {
        chroma_persist_enabled: true,
        kg_aware_retrieval_enabled: true,
      },
      effective: {
        chroma_persist_enabled: true,
        chroma_persist_mode: "disk",
        chroma_persist_path: ".../data/chroma",
        ephemeral_forced: false,
        kg_aware_retrieval_enabled: true,
      },
      restart_required: false,
    });
    render(<Settings />);
    await waitFor(() => expect(api.getRetrievalSettings).toHaveBeenCalled());

    const toggle = await screen.findByRole("switch", {
      name: /use the knowledge graph for candidate discovery/i,
    });
    expect(toggle).toHaveAttribute("aria-checked", "false");
    fireEvent.click(toggle);
    fireEvent.click(screen.getByRole("button", { name: /save retrieval/i }));

    await waitFor(() =>
      expect(api.updateRetrievalSettings).toHaveBeenCalledWith({
        chroma_persist_enabled: true,
        kg_aware_retrieval_enabled: true,
      }),
    );
    expect(await screen.findByRole("status")).toHaveTextContent(/^Saved\.$/);
    expect(screen.getByText(/never replace External search/i)).toBeInTheDocument();
  });

  it("disables Chroma persistence switch when Chroma is not the active backend", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.health).mockResolvedValue({
      status: "ok",
      fts5: true,
      vector_backend: "bruteforce",
      chroma_persist_enabled: true,
      chroma_persist_mode: "disk",
      chroma_persist_path: ".../data/chroma",
      job_backend: "thread",
      redis_reachable: null,
    });
    render(<Settings />);
    await waitFor(() => expect(api.getRetrievalSettings).toHaveBeenCalled());

    expect(await screen.findByRole("switch", { name: /persist chroma vector index on disk/i })).toBeDisabled();
    expect(screen.getByText(/Chroma backend is not active/i)).toBeInTheDocument();
  });

  it("saves only non-empty search-provider fields and reflects updated status", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.saveSearchProviders).mockResolvedValue({
      operation_status: "success",
      revision: 3,
      providers: configured.search_providers!,
      saved: ["brave"],
      rejected: [],
      external_search_service: { status: "ready", runtime_ready: true },
      banner: { code: "external_search_ready", severity: "info", message: "External search providers are available." },
    });

    render(<Settings />);
    await waitFor(() => expect(api.providers).toHaveBeenCalled());

    fireEvent.change(screen.getByPlaceholderText("BSA..."), { target: { value: "BSA-secret" } });

    fireEvent.click(screen.getByRole("button", { name: /save search providers/i }));

    await waitFor(() => expect(api.saveSearchProviders).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(api.saveSearchProviders).mock.calls[0][0];
    expect(payload).toEqual({ expected_revision: 0, providers: { brave: { api_key: "BSA-secret", remove: false } } });
    // Empty fields are not sent.
    expect(payload.providers).not.toHaveProperty("firecrawl");
    await screen.findByText(/External search providers are available/i);
  });

  it("masks already-configured keys with a placeholder and keeps the input blank", async () => {
    vi.mocked(api.providers).mockResolvedValue(configured);
    render(<Settings />);
    await waitFor(() => expect(api.providers).toHaveBeenCalled());

    const masked = screen.getAllByPlaceholderText(/leave blank to keep/i) as HTMLInputElement[];
    expect(masked.length).toBeGreaterThanOrEqual(2); // Brave + Firecrawl configured
    masked.forEach((input) => {
      expect(input.value).toBe("");
      expect(input.type).toBe("password");
    });
  });

  it("keeps a rejected draft while clearing the provider accepted by a partial save", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.saveSearchProviders).mockResolvedValue({
      operation_status: "partial",
      revision: 1,
      providers: configured.search_providers!,
      saved: ["brave"],
      rejected: ["tavily"],
      external_search_service: { status: "warn", runtime_ready: true },
    });
    render(<Settings />);
    await waitFor(() => expect(api.providers).toHaveBeenCalled());
    const brave = screen.getByLabelText("Brave Search API key") as HTMLInputElement;
    const tavily = screen.getByPlaceholderText("tvly-...") as HTMLInputElement;
    fireEvent.change(brave, { target: { value: "brave-draft" } });
    fireEvent.change(tavily, { target: { value: "tavily-draft" } });
    fireEvent.click(screen.getByRole("button", { name: /save search providers/i }));
    await waitFor(() => expect(api.saveSearchProviders).toHaveBeenCalled());
    expect(brave.value).toBe("");
    expect(tavily.value).toBe("tavily-draft");
  });

  it("renders env credentials as read-only and toggles through the state API", async () => {
    const envStatus: ProviderStatus = {
      ...configured,
      search_providers: {
        ...configured.search_providers!,
        exa: {
          configured: true,
          search_ready: true,
          extract_ready: true,
          enabled: true,
          runtime_eligible: true,
          selectable: true,
          credential_source: "env",
          probe_state: "not_tested",
          lamp: "yellow",
        },
      },
    };
    vi.mocked(api.providers).mockResolvedValue(envStatus);
    vi.mocked(api.toggleSearchProvider).mockResolvedValue({
      operation_status: "success",
      revision: 4,
      providers: {
        ...envStatus.search_providers!,
        exa: { ...envStatus.search_providers!.exa, enabled: false, runtime_eligible: false, selectable: false, lamp: "gray" },
      },
      external_search_service: { status: "ready", runtime_ready: true },
    });
    render(<Settings />);
    expect(await screen.findByPlaceholderText("exa-...")).toBeDisabled();
    expect(screen.getByRole("button", { name: /remove exa credential/i })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /disable exa/i }));
    await waitFor(() => expect(api.toggleSearchProvider).toHaveBeenCalledWith({
      expected_revision: 3,
      provider: "exa",
      enabled: false,
    }));
  });

  it("edits and saves deterministic search usage without touching provider keys", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.updateSearchUsage).mockResolvedValue({
      usage_mode: "single",
      primary_provider: "exa",
      fallback_chain: ["tavily", "firecrawl"],
    });
    render(<Settings />);
    await waitFor(() => expect(api.getSearchUsage).toHaveBeenCalled());

    fireEvent.change(screen.getByRole("combobox", { name: /search usage mode/i }), {
      target: { value: "single" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: /primary provider/i }), {
      target: { value: "exa" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^remove brave$/i }));
    fireEvent.click(screen.getByRole("button", { name: /save search usage/i }));

    await waitFor(() =>
      expect(api.updateSearchUsage).toHaveBeenCalledWith({
        usage_mode: "single",
        primary_provider: "exa",
        fallback_chain: ["tavily", "exa", "firecrawl"],
      }),
    );
    expect(api.saveSearchProviders).not.toHaveBeenCalled();
  });

  it("loads post-run notes disabled by default and saves an enabled mode", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.updatePostRunNotes).mockResolvedValue({ enabled_modes: ["deep"] });
    render(<Settings />);
    await waitFor(() => expect(api.getPostRunNotes).toHaveBeenCalled());

    expect(
      screen.getByText(/disabled by default.*library note.*deep\/council reports and citation snapshots/i),
    ).toBeInTheDocument();
    const deep = screen.getByRole("switch", { name: /deep research post-run notes/i });
    const deepTile = screen.getByText("Deep Research").closest("div");
    const fastTile = screen.getByText("Fast Search").closest("div");
    expect(deep).toHaveAttribute("aria-checked", "false");
    expect(deep).toHaveClass(
      "border-text-muted/60",
      "bg-text-muted/40",
      "dark:border-text-muted/45",
      "dark:bg-text-muted/25",
      "shadow-inner",
    );
    expect(deepTile).not.toHaveClass("border-accent");
    fireEvent.click(deep);
    expect(deep).toHaveClass("bg-accent");
    expect(deep).not.toHaveClass("bg-text-muted/40");
    expect(deepTile).toHaveClass("border-accent", "bg-accent-soft", "ring-2", "ring-accent/25");
    expect(fastTile).not.toHaveClass("border-accent");
    fireEvent.click(screen.getByRole("button", { name: /save notes/i }));

    await waitFor(() => expect(api.updatePostRunNotes).toHaveBeenCalledWith({ enabled_modes: ["deep"] }));
    expect(await screen.findByText(/Saved post-run note settings/i)).toBeInTheDocument();
  });

  it("loads and saves the post-run knowledge graph rebuild setting", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.getKnowledgeGraphSettings).mockResolvedValue({
      post_run_rebuild_enabled: true,
      wiki_auto_refresh_after_kg_rebuild: false,
    });
    vi.mocked(api.updateKnowledgeGraphSettings).mockResolvedValue({
      post_run_rebuild_enabled: false,
      wiki_auto_refresh_after_kg_rebuild: true,
    });
    render(<Settings />);
    await waitFor(() => expect(api.getKnowledgeGraphSettings).toHaveBeenCalled());

    const toggle = screen.getByRole("switch", {
      name: /refresh knowledge graph after completed research/i,
    });
    expect(toggle).toHaveAttribute("aria-checked", "true");
    fireEvent.click(toggle);
    fireEvent.click(
      screen.getByRole("switch", {
        name: /refresh wiki map after knowledge graph rebuild/i,
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: /save graph/i }));

    await waitFor(() =>
      expect(api.updateKnowledgeGraphSettings).toHaveBeenCalledWith({
        post_run_rebuild_enabled: false,
        wiki_auto_refresh_after_kg_rebuild: true,
      }),
    );
    expect(await screen.findByText(/Saved knowledge graph settings/i)).toBeInTheDocument();
  });

  it("clears retrieval cache only after confirmation", async () => {
    vi.mocked(api.providers).mockResolvedValue(unconfigured);
    vi.mocked(api.purgeCache).mockResolvedValue({ purged_count: 3, scope: "all" });
    render(<Settings />);
    await waitFor(() => expect(api.health).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /clear retrieval cache/i }));

    expect(api.purgeCache).not.toHaveBeenCalled();
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveTextContent(/clear retrieval cache/i);
    expect(dialog).toHaveTextContent(/does not delete sessions, runs, memories, library items, or byok settings/i);

    fireEvent.click(screen.getByRole("button", { name: /clear cache/i }));

    await waitFor(() => expect(api.purgeCache).toHaveBeenCalledWith({ scope: "all" }));
    expect(await screen.findByText(/Cleared 3 cache entries/i)).toBeInTheDocument();
  });
});
