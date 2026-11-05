import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, render, screen, waitFor, fireEvent } from "@testing-library/react";
import type { ReactElement } from "react";
import { Composer } from "./Composer";
import { api } from "@/lib/api";
import type { ModelKind, ModelProfile } from "@/lib/types";

function model(id: string, kind: ModelKind = "chat"): ModelProfile {
  return {
    id,
    alias: "",
    kind,
    context_window: 128000,
    capabilities: { vision: false, ocr: false, web_search: false, thinking: false },
    thinking_budget: null,
    temperature: null,
    dimensions: null,
    enabled: true,
  };
}

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      enhancePrompt: vi.fn(),
      estimateRunCost: vi.fn(),
      getSearchUsage: vi.fn(),
      llmRegistry: vi.fn(),
    },
  };
});

const defaultEstimate = {
  mode: "fast",
  searches: 1,
  search_usd_low: 0.005,
  search_usd_high: 0.005,
  search_credits_low: null,
  search_credits_high: null,
  llm_input_tokens_est: 2000,
  llm_output_tokens_est: 700,
  llm_usd_low: 0.001,
  llm_usd_high: 0.001,
  total_usd_low: 0.006,
  total_usd_high: 0.006,
  confidence: "low",
  pricing_basis: {},
  disclaimers: ["Estimate only."],
};

const registry = {
  providers: [
    {
      id: "openai",
      label: "OpenAI",
      type: "openai",
      base_url: "",
      enabled: true,
      has_key: true,
      models: [model("gpt-4o"), model("gpt-4o-mini"), model("text-embedding-3-small", "embed")],
    },
  ],
  roles: { default: "openai:gpt-4o", embed: "openai:text-embedding-3-small", memory: "", council: [] },
};

async function renderComposer(ui: ReactElement) {
  const result = render(ui);
  await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());
  return result;
}

describe("Composer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.llmRegistry).mockResolvedValue(registry);
    vi.mocked(api.getSearchUsage).mockResolvedValue({
      usage_mode: "auto",
      fallback_chain: ["brave", "tavily", "exa", "firecrawl"],
    });
    vi.mocked(api.estimateRunCost).mockImplementation(() => new Promise(() => undefined));
  });

  it("submits the trimmed query", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "  hello  " } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("hello", undefined);
  });

  it("enhances the prompt and replaces the textarea value", async () => {
    vi.mocked(api.enhancePrompt).mockResolvedValue({ enhanced: "A sharper question?" });
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    const textarea = screen.getByLabelText("Research question") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "vague q" } });
    fireEvent.click(screen.getByLabelText("Enhance prompt"));
    await waitFor(() => expect(api.enhancePrompt).toHaveBeenCalledWith("vague q"));
    await waitFor(() => expect(textarea.value).toBe("A sharper question?"));
  });

  it("passes budget overrides from the options popover", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Max searches"), { target: { value: "5" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { budget: { max_searches: 5 } });
  });

  it("passes a per-run model override from the registry", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "openai:gpt-4o-mini" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { llm_model: "openai:gpt-4o-mini" });
  });

  it("passes per-run temperature and thinking depth overrides", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Temperature"), { target: { value: "0.8" } });
    fireEvent.change(screen.getByLabelText("Thinking depth"), { target: { value: "high" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { temperature: 0.8, thinking_effort: "high" });
  });

  it("excludes embedding models from the chat model selector", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.click(screen.getByLabelText("Options"));
    const select = screen.getByLabelText("Model") as HTMLSelectElement;
    const optionValues = Array.from(select.options).map((o) => o.value);
    expect(optionValues).toContain("openai:gpt-4o");
    expect(optionValues).not.toContain("openai:text-embedding-3-small");
  });

  it("previews the mode budget and reflects search overrides live", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="fast" />);
    const bar = screen.getByLabelText("Budget preview");
    // fast preset: 1 search.
    expect(bar).toHaveTextContent("Searches");
    expect(bar).toHaveTextContent("1");
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Max searches"), { target: { value: "5" } });
    expect(screen.getByLabelText("Budget preview")).toHaveTextContent("5");
  });

  it("includes a Home custom search strategy in the first run payload", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="fast" />);

    fireEvent.click(screen.getByRole("button", { name: /options/i }));
    fireEvent.click(screen.getByRole("button", { name: /^show$/i }));
    fireEvent.click(screen.getByRole("button", { name: /customize this session/i }));
    fireEvent.change(screen.getByRole("textbox", { name: /research question/i }), {
      target: { value: "provider strategy" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(onSubmit).toHaveBeenCalledWith(
      "provider strategy",
      expect.objectContaining({
        search_usage_policy: {
          usage_mode: "auto",
          fallback_chain: ["brave", "tavily", "exa", "firecrawl"],
        },
      }),
    );
  });

  it("keeps the active search strategy visible while advanced controls are collapsed", async () => {
    await renderComposer(<Composer onSubmit={vi.fn()} mode="fast" initialSearchUsagePolicy={null} />);

    expect(screen.getByLabelText("Search strategy summary")).toHaveTextContent(/global default/i);
    expect(screen.queryByTestId("search-usage-editor")).not.toBeInTheDocument();
  });

  it("blocks a session run while a strategy change is unsaved", async () => {
    const onSubmit = vi.fn();
    const persist = vi.fn().mockRejectedValue(new Error("patch failed"));
    await renderComposer(
      <Composer
        onSubmit={onSubmit}
        mode="fast"
        initialSearchUsagePolicy={null}
        onSearchUsagePolicyChange={persist}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /options/i }));
    fireEvent.click(screen.getByRole("button", { name: /^show$/i }));
    fireEvent.click(screen.getByRole("button", { name: /customize this session/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/not saved/i);
    fireEvent.change(screen.getByRole("textbox", { name: /research question/i }), {
      target: { value: "must not run" },
    });

    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("keeps the newest local strategy while an older save response is pending", async () => {
    let resolveSave!: () => void;
    const persist = vi.fn(() => new Promise<void>((resolve) => {
      resolveSave = resolve;
    }));
    const initial = {
      usage_mode: "auto" as const,
      fallback_chain: ["brave", "tavily", "exa", "firecrawl"] as const,
    };
    const view = await renderComposer(
      <Composer
        onSubmit={vi.fn()}
        mode="fast"
        initialSearchUsagePolicy={{ ...initial, fallback_chain: [...initial.fallback_chain] }}
        onSearchUsagePolicyChange={persist}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /options/i }));
    fireEvent.click(screen.getByRole("button", { name: /^show$/i }));
    fireEvent.click(screen.getByRole("button", { name: "Move tavily up" }));
    await waitFor(() => expect(persist).toHaveBeenCalledWith({
        usage_mode: "auto",
        fallback_chain: ["tavily", "brave", "exa", "firecrawl"],
    }));

    view.rerender(
      <Composer
        onSubmit={vi.fn()}
        mode="fast"
        initialSearchUsagePolicy={{ ...initial, fallback_chain: [...initial.fallback_chain] }}
        onSearchUsagePolicyChange={persist}
      />,
    );
    const visible = Array.from(
      screen.getByRole("list", { name: "External fallback order" }).querySelectorAll("li"),
    ).map((item) => item.textContent?.trim().split(/\s+/)[0].toLowerCase());
    expect(visible).toEqual(["tavily", "brave", "exa", "firecrawl"]);

    await act(async () => resolveSave());
    await waitFor(() => expect(screen.getByLabelText("Research question")).toBeEnabled());
  });

  it("loads the provider cost estimate for the selected mode", async () => {
    vi.mocked(api.estimateRunCost).mockResolvedValue(defaultEstimate);
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="fast" />);

    await waitFor(() => expect(api.estimateRunCost).toHaveBeenCalledWith({ mode: "fast" }));
    await waitFor(() => expect(screen.getByLabelText("Budget preview")).toHaveTextContent("$0.0060"));
    expect(screen.getByLabelText("Budget preview")).toHaveTextContent("~2,000 in / ~700 out");
  });

  it("scales the council budget preview with member count", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="council" />);
    const bar = screen.getByLabelText("Budget preview");
    // council preset output 1400 × default size 3 = 4,200 tokens.
    expect(bar).toHaveTextContent("4,200");
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Council member count"), { target: { value: "5" } });
    // 1400 × 5 = 7,000.
    expect(screen.getByLabelText("Budget preview")).toHaveTextContent("7,000");
  });

  it("passes council size and candidate members when in council mode", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="council" />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Council member count"), { target: { value: "4" } });
    fireEvent.click(screen.getByLabelText("council member openai:gpt-4o"));
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", {
      council_size: 4,
      council_models: ["openai:gpt-4o"],
      council_strategy: "judge_synthesis",
    });
  });

  it("selects one council strategy at a time and submits parallel_vote", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="council" />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    const parallel = screen.getByLabelText("Council strategy Parallel vote");
    const judge = screen.getByLabelText("Council strategy Judge synthesis");

    expect(judge).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(parallel);
    expect(parallel).toHaveAttribute("aria-pressed", "true");
    expect(judge).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { council_size: 3, council_strategy: "parallel_vote" });
  });

  it("passes council_strategy debate_once when the critique card is selected", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="council" />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.click(screen.getByLabelText("Council strategy One critique round"));
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { council_size: 3, council_strategy: "debate_once" });
  });

  it("sends memory_policy=workspace when the workspace memory toggle is on (DR-ITER-054)", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.click(screen.getByLabelText("Use workspace memory"));
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { memory_policy: "workspace" });
  });

  it("defaults the workspace memory toggle on for follow-ups", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} defaultUseMemory />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { memory_policy: "workspace" });
  });

  it("omits memory_policy when the toggle stays off", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", undefined);
  });

  it("omits library_scope when Library context stays off", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", undefined);
  });

  it("passes notes Library scope when selected", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Library context"), { target: { value: "notes" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { library_scope: "notes" });
  });

  it("passes reports Library scope when selected", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Library context"), { target: { value: "reports" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { library_scope: "reports" });
  });

  it("passes all Library scope when selected", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Library context"), { target: { value: "all" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { library_scope: "all" });
  });

  it("keeps advanced search collapsed and omits empty search_options", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="multiple" />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));

    expect(screen.getByText(/Advanced search/i)).toBeInTheDocument();
    expect(screen.queryByLabelText("Include domains")).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", undefined);
  });

  it("submits advanced search options when fields are filled", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="multiple" />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    fireEvent.change(screen.getByLabelText("Include domains"), { target: { value: "example.com, docs.example.com" } });
    fireEvent.change(screen.getByLabelText("Exclude domains"), { target: { value: "spam.example" } });
    fireEvent.change(screen.getByLabelText("Search freshness"), { target: { value: "pw" } });
    fireEvent.change(screen.getByLabelText("Extra subqueries"), { target: { value: "alpha\nbeta" } });
    fireEvent.change(screen.getByLabelText("Search country"), { target: { value: "SG" } });
    fireEvent.change(screen.getByLabelText("Search language"), { target: { value: "zh-hans" } });
    fireEvent.change(screen.getByLabelText("搜索结果界面语言"), { target: { value: "zh-Hans-SG" } });

    fireEvent.click(screen.getByLabelText("Send"));

    expect(onSubmit).toHaveBeenCalledWith("q", {
      search_options: {
        include_domains: ["example.com", "docs.example.com"],
        exclude_domains: ["spam.example"],
        freshness: "pw",
        extra_subqueries: ["alpha", "beta"],
        country: "SG",
        search_lang: "zh-hans",
        ui_language: "zh-Hans-SG",
      },
    });
  });

  it("includes advanced search and model choices in the cost estimate payload", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="multiple" />);
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "openai:gpt-4o-mini" } });
    fireEvent.change(screen.getByLabelText("Max searches"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    fireEvent.change(screen.getByLabelText("Include domains"), { target: { value: "example.com" } });
    fireEvent.change(screen.getByLabelText("Search freshness"), { target: { value: "pd" } });

    await waitFor(() =>
      expect(api.estimateRunCost).toHaveBeenLastCalledWith({
        mode: "multiple",
        budget: { max_searches: 3 },
        llm_model: "openai:gpt-4o-mini",
        search_options: { include_domains: ["example.com"], freshness: "pd" },
      }),
    );
  });

  it("omits context_budget when the global setting is selected (DR-ITER-057)", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", undefined);
  });

  it("sends a per-run context_budget preset override (DR-ITER-057)", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText("Research question"), { target: { value: "q" } });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.change(screen.getByLabelText("Context budget"), { target: { value: "followup" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSubmit).toHaveBeenCalledWith("q", { context_budget: { preset: "followup" } });
  });

  it("updates the council budget preview when a strategy card is selected", async () => {
    const onSubmit = vi.fn();
    await renderComposer(<Composer onSubmit={onSubmit} mode="council" />);
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.click(screen.getByLabelText("Council strategy Parallel vote"));
    // 1400 × 3 × 0.85 = 3,570.
    expect(screen.getByLabelText("Budget preview")).toHaveTextContent("3,570");
    fireEvent.click(screen.getByLabelText("Council strategy One critique round"));
    // 1400 × 3 × 1.3 = 5,460.
    expect(screen.getByLabelText("Budget preview")).toHaveTextContent("5,460");
  });
});
