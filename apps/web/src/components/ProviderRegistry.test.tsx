import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import { ProviderRegistry } from "./ProviderRegistry";
import { api } from "@/lib/api";
import type { LLMRegistryStatus, ModelProfile } from "@/lib/types";

function model(id: string, kind: "chat" | "embed" = "chat"): ModelProfile {
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
      llmRegistry: vi.fn(),
      upsertLlmProvider: vi.fn(),
      patchLlmProvider: vi.fn(),
      removeLlmProvider: vi.fn(),
      setLlmRoles: vi.fn(),
      getMemorySettings: vi.fn(),
      updateMemorySettings: vi.fn(),
      probeMemoryFormat: vi.fn(),
      testLlm: vi.fn(),
      listProviderModels: vi.fn(),
      createProviderModels: vi.fn(),
      deleteProviderModel: vi.fn(),
    },
  };
});

const emptyRegistry: LLMRegistryStatus = { providers: [], roles: { default: "", embed: "", memory: "", council: [] } };

const oneProvider: LLMRegistryStatus = {
  providers: [
    {
      id: "openai",
      label: "OpenAI",
      type: "openai",
      base_url: "https://api.openai.com/v1",
      models: [model("gpt-4o-mini")],
      enabled: true,
      has_key: true,
    },
  ],
  roles: { default: "openai:gpt-4o-mini", embed: "", memory: "", council: ["openai:gpt-4o-mini"] },
};

describe("ProviderRegistry", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getMemorySettings).mockResolvedValue({
      post_turn_memory_refresh_enabled: true,
      workspace_memory_refresh_strategy: "incremental",
      memory_thinking_enabled: false,
      format_probe: null,
    });
    vi.mocked(api.updateMemorySettings).mockResolvedValue({
      post_turn_memory_refresh_enabled: true,
      workspace_memory_refresh_strategy: "incremental",
      memory_thinking_enabled: true,
      format_probe: null,
    });
  });

  it("lists configured providers in a table with key state", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);
    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: /toggle openai/i })).toBeInTheDocument();
  });

  it("validates and creates a provider via the dialog", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(emptyRegistry);
    vi.mocked(api.upsertLlmProvider).mockResolvedValue(oneProvider);

    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /add provider/i }));
    const dialog = await screen.findByRole("dialog");

    // Submit empty → validation blocks the call.
    fireEvent.click(within(dialog).getByRole("button", { name: /add provider/i }));
    expect(await within(dialog).findByText(/provider id is required/i)).toBeInTheDocument();
    expect(api.upsertLlmProvider).not.toHaveBeenCalled();

    fireEvent.change(within(dialog).getByLabelText(/provider id/i), { target: { value: "openai" } });
    fireEvent.change(within(dialog).getByLabelText(/base url/i), { target: { value: "https://api.openai.com/v1" } });
    fireEvent.change(within(dialog).getByLabelText("API key"), { target: { value: "sk-secret" } });
    fireEvent.click(within(dialog).getByRole("button", { name: /add provider/i }));

    await waitFor(() => expect(api.upsertLlmProvider).toHaveBeenCalled());
    const payload = vi.mocked(api.upsertLlmProvider).mock.calls[0][0];
    expect(payload.id).toBe("openai");
    expect(payload.base_url).toBe("https://api.openai.com/v1");
    expect(payload.api_key).toBe("sk-secret");
    expect(payload.models).toEqual([]);
  });

  it("edits an existing provider via PATCH", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);
    vi.mocked(api.patchLlmProvider).mockResolvedValue(oneProvider);
    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /edit openai/i }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/a key is set/i)).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: /reveal api key/i })).not.toBeInTheDocument();
    fireEvent.change(within(dialog).getByLabelText(/label/i), { target: { value: "OpenAI Prod" } });
    fireEvent.click(within(dialog).getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(api.patchLlmProvider).toHaveBeenCalled());
    const [id, patch] = vi.mocked(api.patchLlmProvider).mock.calls[0];
    expect(id).toBe("openai");
    expect(patch.label).toBe("OpenAI Prod");
    expect(patch.api_key).toBeUndefined();
  });

  it("replaces an existing provider key without revealing the saved value", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);
    vi.mocked(api.patchLlmProvider).mockResolvedValue(oneProvider);
    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /edit openai/i }));
    const dialog = await screen.findByRole("dialog");
    const keyInput = within(dialog).getByLabelText("API key");
    expect(keyInput).toHaveValue("");
    fireEvent.change(keyInput, { target: { value: "sk-replacement" } });
    fireEvent.click(within(dialog).getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(api.patchLlmProvider).toHaveBeenCalled());
    expect(vi.mocked(api.patchLlmProvider).mock.calls[0][1].api_key).toBe("sk-replacement");
  });

  it("toggles a provider's enabled state", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);
    vi.mocked(api.patchLlmProvider).mockResolvedValue(oneProvider);
    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("switch", { name: /toggle openai/i }));
    await waitFor(() => expect(api.patchLlmProvider).toHaveBeenCalledWith("openai", { enabled: false }));
  });

  it("confirms before deleting a provider", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);
    vi.mocked(api.removeLlmProvider).mockResolvedValue(emptyRegistry);
    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /delete openai/i }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: /^delete$/i }));
    await waitFor(() => expect(api.removeLlmProvider).toHaveBeenCalledWith("openai"));
  });

  it("runs a provider-level test (GET models) and shows the catalog count", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);
    vi.mocked(api.testLlm).mockResolvedValue({
      ok: true,
      status: "reachable",
      latency_ms: 42,
      model: "",
      endpoint_host: "api.openai.com",
      error_message: "",
      models_count: 7,
    });
    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /test openai/i }));
    // Provider-row test sends the default probe (no model) -> backend lists models.
    await waitFor(() => expect(api.testLlm).toHaveBeenCalledWith({ provider_id: "openai" }));
    expect(await screen.findByText(/models=7/i)).toBeInTheDocument();
  });

  it("saves role bindings", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);
    vi.mocked(api.setLlmRoles).mockResolvedValue(oneProvider);
    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText(/default model/i), { target: { value: "openai:gpt-4o-mini" } });
    fireEvent.click(screen.getByRole("button", { name: /save roles/i }));
    await waitFor(() => expect(api.setLlmRoles).toHaveBeenCalled());
    expect(vi.mocked(api.setLlmRoles).mock.calls[0][0]).toEqual(
      expect.objectContaining({ default: "openai:gpt-4o-mini" }),
    );
  });

  it("saves the memory processing role", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);
    vi.mocked(api.setLlmRoles).mockResolvedValue({
      ...oneProvider,
      roles: { ...oneProvider.roles, memory: "openai:gpt-4o-mini" },
    });
    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText(/memory processing model/i), {
      target: { value: "openai:gpt-4o-mini" },
    });
    expect(screen.getByText(/non-reasoning small model/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /save roles/i }));

    await waitFor(() => expect(api.setLlmRoles).toHaveBeenCalled());
    expect(vi.mocked(api.setLlmRoles).mock.calls[0][0]).toEqual(
      expect.objectContaining({ memory: "openai:gpt-4o-mini" }),
    );
  });

  it("persists the memory thinking toggle", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue(oneProvider);

    render(<ProviderRegistry />);
    await waitFor(() => expect(api.getMemorySettings).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("switch", { name: /memory thinking/i }));

    await waitFor(() => expect(api.updateMemorySettings).toHaveBeenCalledWith({ memory_thinking_enabled: true }));
    expect(await screen.findByText("On")).toBeInTheDocument();
    expect(screen.getByText(/format tests keep thinking off/i)).toBeInTheDocument();
  });

  it("tests the selected memory format and shows the probe badge", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue({
      ...oneProvider,
      roles: { ...oneProvider.roles, memory: "openai:gpt-4o-mini" },
    });
    vi.mocked(api.setLlmRoles).mockResolvedValue({
      ...oneProvider,
      roles: { ...oneProvider.roles, memory: "openai:gpt-4o-mini" },
    });
    vi.mocked(api.probeMemoryFormat).mockResolvedValue({
      status: "caution",
      provider_ref: "openai:gpt-4o-mini",
      capability: "json_object",
      structured_strategy: "json_object",
      checked_at: 1,
      diagnostics: { response_shape: "json_object" },
    });

    render(<ProviderRegistry />);
    await waitFor(() => expect(api.llmRegistry).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /test memory format/i }));

    await waitFor(() => expect(api.probeMemoryFormat).toHaveBeenCalled());
    expect(await screen.findByText("Caution")).toBeInTheDocument();
  });

  it("shows stale memory format state from settings", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValue({
      ...oneProvider,
      roles: { ...oneProvider.roles, memory: "openai:gpt-4o-mini" },
    });
    vi.mocked(api.getMemorySettings).mockResolvedValue({
      post_turn_memory_refresh_enabled: true,
      workspace_memory_refresh_strategy: "incremental",
      memory_thinking_enabled: false,
      format_probe: {
        status: "passed",
        provider_ref: "openai:gpt-4o-mini",
        capability: "json_schema_strict",
        structured_strategy: "json_schema_strict",
        checked_at: 1,
        stale: true,
        stale_reason: "memory_role_changed",
        diagnostics: { response_shape: "json_object" },
      },
    });

    render(<ProviderRegistry />);

    expect(await screen.findByText("Stale")).toBeInTheDocument();
    expect(screen.getByText(/test memory format again/i)).toBeInTheDocument();
  });
});
