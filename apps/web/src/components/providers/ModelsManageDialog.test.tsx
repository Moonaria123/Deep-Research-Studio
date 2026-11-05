import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { ModelsManageDialog } from "./ModelsManageDialog";
import { Toaster } from "@/components/ui/toast";
import { api } from "@/lib/api";
import type { LLMProviderProfile, ModelListResult, ModelProfile } from "@/lib/types";

function model(id: string, overrides: Partial<ModelProfile> = {}): ModelProfile {
  return {
    id,
    alias: "",
    kind: "chat",
    context_window: 128000,
    capabilities: { vision: false, ocr: false, web_search: false, thinking: false },
    thinking_budget: null,
    temperature: null,
    dimensions: null,
    enabled: true,
    ...overrides,
  };
}

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      listProviderModels: vi.fn(),
      createProviderModels: vi.fn(),
      updateProviderModel: vi.fn(),
      deleteProviderModel: vi.fn(),
      fetchModels: vi.fn(),
      testLlm: vi.fn(),
    },
  };
});

const provider: LLMProviderProfile = {
  id: "openai",
  label: "OpenAI",
  type: "openai",
  base_url: "https://api.openai.com/v1",
  models: [model("gpt-4o-mini")],
  enabled: true,
  has_key: true,
};

function upstream(models: string[]): ModelListResult {
  return {
    ok: true,
    status: "ok",
    models,
    latency_ms: 12,
    endpoint_host: "api.openai.com",
    error_message: "",
  };
}

describe("ModelsManageDialog", () => {
  beforeEach(() => vi.clearAllMocks());

  it("lists existing models on open", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [model("gpt-4o-mini")] });
    render(<ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={() => {}} />);
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalledWith("openai"));
    expect(await screen.findByText("gpt-4o-mini")).toBeInTheDocument();
  });

  it("adds a manual model with full parameters", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [] });
    vi.mocked(api.createProviderModels).mockResolvedValue({
      id: "openai",
      added: 1,
      models: [model("custom-model")],
    });
    const onChanged = vi.fn();
    render(<ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={onChanged} />);
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText("Add model id"), { target: { value: "custom-model" } });
    fireEvent.change(screen.getByLabelText("Manual alias"), { target: { value: "Custom" } });
    fireEvent.change(screen.getByLabelText("Manual context window"), { target: { value: "32000" } });
    fireEvent.click(screen.getByRole("switch", { name: /manual vision capability/i }));
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(api.createProviderModels).toHaveBeenCalled());
    const [pid, body] = vi.mocked(api.createProviderModels).mock.calls[0];
    expect(pid).toBe("openai");
    expect(body.models[0]).toMatchObject({
      id: "custom-model",
      alias: "Custom",
      context_window: 32000,
      capabilities: { vision: true },
    });
    expect(onChanged).toHaveBeenCalled();
  });

  it("adds a manual model with temperature and thinking depth defaults", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [] });
    vi.mocked(api.createProviderModels).mockResolvedValue({
      id: "openai",
      added: 1,
      models: [model("deep-model")],
    });
    render(<ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={() => {}} />);
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText("Add model id"), { target: { value: "deep-model" } });
    fireEvent.change(screen.getByLabelText("Manual default temperature"), { target: { value: "0.5" } });
    fireEvent.change(screen.getByLabelText("Manual thinking depth"), { target: { value: "medium" } });
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(api.createProviderModels).toHaveBeenCalled());
    const [, body] = vi.mocked(api.createProviderModels).mock.calls[0];
    expect(body.models[0]).toMatchObject({
      id: "deep-model",
      temperature: 0.5,
      thinking_budget: 4096,
      capabilities: { thinking: true },
    });
  });

  it("fetches upstream and adds a single model selected from the dropdown", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [model("gpt-4o-mini")] });
    vi.mocked(api.fetchModels).mockResolvedValue(upstream(["gpt-4o-mini", "gpt-4o", "o3-mini"]));
    vi.mocked(api.createProviderModels).mockResolvedValue({
      id: "openai",
      added: 1,
      models: [model("gpt-4o-mini"), model("gpt-4o")],
    });
    render(<ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={() => {}} />);
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /fetch available models/i }));
    await waitFor(() => expect(api.fetchModels).toHaveBeenCalledWith({ provider_id: "openai" }));

    // Already-configured ids are excluded from the dropdown options.
    expect(screen.queryByRole("option", { name: "gpt-4o-mini" })).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: "gpt-4o" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Upstream models"), { target: { value: "gpt-4o" } });
    fireEvent.click(screen.getByRole("button", { name: /add selected model/i }));

    await waitFor(() =>
      expect(api.createProviderModels).toHaveBeenCalledWith("openai", { models: [{ id: "gpt-4o" }] }),
    );
  });

  it("filters the upstream dropdown to matching ids", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [model("gpt-4o-mini")] });
    vi.mocked(api.fetchModels).mockResolvedValue(upstream(["gpt-4o-mini", "gpt-4o", "o3-mini"]));
    render(<ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={() => {}} />);
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /fetch available models/i }));
    await waitFor(() => expect(api.fetchModels).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText(/filter upstream models/i), { target: { value: "o3" } });
    expect(screen.getByRole("option", { name: "o3-mini" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "gpt-4o" })).not.toBeInTheDocument();
  });

  it("runs a per-model chat probe via testLlm", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [model("gpt-4o-mini")] });
    vi.mocked(api.testLlm).mockResolvedValue({
      ok: true,
      status: "reachable",
      latency_ms: 31,
      model: "gpt-4o-mini",
      endpoint_host: "api.openai.com",
      error_message: "",
    });
    render(<ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={() => {}} />);
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /test chat gpt-4o-mini/i }));
    await waitFor(() =>
      expect(api.testLlm).toHaveBeenCalledWith({
        provider_id: "openai",
        probe: "chat",
        model: "gpt-4o-mini",
      }),
    );
    expect(await screen.findByText(/reachable/i)).toBeInTheDocument();
  });

  it("surfaces an upstream fetch error", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [] });
    vi.mocked(api.fetchModels).mockResolvedValue({
      ok: false,
      status: "rate_limited",
      models: [],
      latency_ms: null,
      endpoint_host: "api.openai.com",
      error_message: "Rate limited by provider",
    });
    render(
      <>
        <ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={() => {}} />
        <Toaster />
      </>,
    );
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /fetch available models/i }));
    expect(await screen.findByText(/rate limited by provider/i)).toBeInTheDocument();
    expect(api.createProviderModels).not.toHaveBeenCalled();
  });

  it("edits a model's alias and capabilities inline", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [model("gpt-4o-mini")] });
    vi.mocked(api.updateProviderModel).mockResolvedValue({
      id: "openai",
      models: [
        model("gpt-4o-mini", {
          alias: "Mini",
          capabilities: { vision: true, ocr: false, web_search: false, thinking: false },
        }),
      ],
    });
    render(<ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={() => {}} />);
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /edit model gpt-4o-mini/i }));
    fireEvent.change(screen.getByLabelText("Alias"), { target: { value: "Mini" } });
    fireEvent.click(screen.getByRole("switch", { name: /^vision capability/i }));
    fireEvent.change(screen.getByLabelText("Default temperature"), { target: { value: "0.3" } });
    fireEvent.change(screen.getByLabelText("Thinking depth"), { target: { value: "low" } });
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => expect(api.updateProviderModel).toHaveBeenCalled());
    const [pid, mid, body] = vi.mocked(api.updateProviderModel).mock.calls[0];
    expect(pid).toBe("openai");
    expect(mid).toBe("gpt-4o-mini");
    expect(body.alias).toBe("Mini");
    expect(body.capabilities?.vision).toBe(true);
    expect(body.temperature).toBe(0.3);
    expect(body.thinking_budget).toBe(1024);
  });

  it("deletes a model", async () => {
    vi.mocked(api.listProviderModels).mockResolvedValue({ id: "openai", models: [model("gpt-4o-mini")] });
    vi.mocked(api.deleteProviderModel).mockResolvedValue({
      providers: [],
      roles: { default: "", embed: "", memory: "", council: [] },
    });
    const onChanged = vi.fn();
    render(<ModelsManageDialog open provider={provider} onOpenChange={() => {}} onChanged={onChanged} />);
    await waitFor(() => expect(api.listProviderModels).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /remove model gpt-4o-mini/i }));
    await waitFor(() => expect(api.deleteProviderModel).toHaveBeenCalledWith("openai", "gpt-4o-mini"));
    expect(onChanged).toHaveBeenCalled();
  });
});
