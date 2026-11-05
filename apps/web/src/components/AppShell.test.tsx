import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { AppShell, useProvider, useProviderActions } from "./AppShell";
import { api } from "@/lib/api";
import type { ProviderStatus } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  api: {
    providers: vi.fn(() => new Promise(() => undefined)),
  },
}));

vi.mock("./ThemeToggle", () => ({
  ThemeToggle: () => <button type="button">Theme</button>,
}));

function stubLocalStorage() {
  const store = new Map<string, string>();
  const storage = {
    getItem: vi.fn((key: string) => store.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => store.set(key, value)),
    removeItem: vi.fn((key: string) => store.delete(key)),
    clear: vi.fn(() => store.clear()),
  };
  vi.stubGlobal("localStorage", storage);
}

function renderShell(indexElement = <div data-testid="page-content">Page</div>) {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={indexElement} />
          <Route path="/history" element={<div data-testid="history-content">History</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

const providerStatus = (revision: number, enabled: boolean): ProviderStatus => ({
  llm_base_url: "",
  llm_model: "",
  has_llm_key: false,
  brave_ready: true,
  firecrawl_ready: false,
  embed_model: "",
  ollama_base_url: "",
  provider_config_revision: revision,
  search_providers: {
    brave: {
      configured: true,
      search_ready: true,
      extract_ready: false,
      enabled,
      runtime_eligible: enabled,
      selectable: enabled,
    },
    tavily: { configured: false, search_ready: false, extract_ready: false, enabled: true },
    exa: { configured: false, search_ready: false, extract_ready: false, enabled: true },
    firecrawl: { configured: false, search_ready: false, extract_ready: false, enabled: true },
  },
  external_search_service: {
    status: enabled ? "ready" : "unavailable",
    runtime_ready: enabled,
  },
});

function ProviderRevisionProbe() {
  const provider = useProvider();
  const { publishProviderStatus } = useProviderActions();
  return (
    <div>
      <span data-testid="provider-revision">{provider?.provider_config_revision ?? "none"}</span>
      <span data-testid="brave-enabled">{String(provider?.search_providers?.brave.enabled)}</span>
      <button type="button" onClick={() => publishProviderStatus(providerStatus(2, false))}>Publish revision 2</button>
      <button type="button" onClick={() => publishProviderStatus(providerStatus(1, true))}>Publish stale revision 1</button>
    </div>
  );
}

function mockCompactShell(matches: boolean) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: query === "(max-width: 767px)" ? matches : false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}

describe("AppShell", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.providers).mockImplementation(() => new Promise(() => undefined));
    stubLocalStorage();
    mockCompactShell(false);
  });

  it("publishes provider state across routes without allowing an older revision to overwrite it", async () => {
    vi.mocked(api.providers).mockResolvedValue(providerStatus(1, true));
    renderShell(<ProviderRevisionProbe />);

    await waitFor(() => expect(screen.getByTestId("provider-revision")).toHaveTextContent("1"));
    fireEvent.click(screen.getByRole("button", { name: /publish revision 2/i }));
    expect(screen.getByTestId("provider-revision")).toHaveTextContent("2");
    expect(screen.getByTestId("brave-enabled")).toHaveTextContent("false");
    expect(screen.getByText(/External search unavailable/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /publish stale revision 1/i }));
    expect(screen.getByTestId("provider-revision")).toHaveTextContent("2");
    expect(screen.getByTestId("brave-enabled")).toHaveTextContent("false");
  });

  it("keeps the desktop sidebar docked outside compact mode", () => {
    renderShell();

    expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Open navigation" })).toBeNull();
    expect(screen.getByTestId("app-shell-main")).toHaveClass("min-w-0");
    expect(screen.getByTestId("page-content")).toBeInTheDocument();
  });

  it("uses a compact drawer that closes from backdrop, escape, and navigation", () => {
    mockCompactShell(true);
    renderShell();

    expect(screen.queryByRole("navigation", { name: "Primary" })).toBeNull();
    const menu = screen.getByRole("button", { name: "Open navigation" });
    fireEvent.click(menu);

    const dialog = screen.getByRole("dialog", { name: "Primary navigation" });
    expect(within(dialog).getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "Collapse sidebar" })).toBeNull();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Primary navigation" })).toBeNull();
    expect(menu).toHaveFocus();

    fireEvent.click(menu);
    fireEvent.click(screen.getByTestId("app-shell-nav-backdrop"));
    expect(screen.queryByRole("dialog", { name: "Primary navigation" })).toBeNull();

    fireEvent.click(menu);
    fireEvent.click(screen.getByRole("link", { name: "History" }));
    expect(screen.queryByRole("dialog", { name: "Primary navigation" })).toBeNull();
  });

  it("does not mutate the desktop sidebar collapse preference in compact mode", () => {
    mockCompactShell(true);
    localStorage.setItem("deepresearch-sidebar-collapsed", "1");
    renderShell();

    fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
    expect(screen.getByRole("dialog", { name: "Primary navigation" })).toBeInTheDocument();
    expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close navigation" }));
    expect(localStorage.getItem("deepresearch-sidebar-collapsed")).toBe("1");
  });
});
