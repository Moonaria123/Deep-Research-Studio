import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { SessionThread } from "./SessionThread";
import { api, ApiError } from "@/lib/api";

const navigate = vi.fn();
const localStorageStore = new Map<string, string>();

function installLocalStorageMock() {
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => localStorageStore.get(key) ?? null,
      setItem: (key: string, value: string) => {
        localStorageStore.set(key, value);
      },
      removeItem: (key: string) => {
        localStorageStore.delete(key);
      },
      clear: () => localStorageStore.clear(),
    },
  });
}
vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>();
  return { ...actual, useNavigate: () => navigate, useParams: () => ({ sessionId: "s1" }) };
});
vi.mock("@/pages/RunView", () => ({
  RunPanel: ({ runId }: { runId?: string }) => <div data-testid="run-panel">{runId ?? "empty"}</div>,
}));
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      listMessages: vi.fn(),
      listSessions: vi.fn(),
      patchSession: vi.fn(),
      createRun: vi.fn(),
      searchHistory: vi.fn().mockResolvedValue({ results: [], scope: "intra" }),
      getSearchUsage: vi.fn().mockResolvedValue({
        usage_mode: "auto",
        fallback_chain: ["brave", "tavily", "exa", "firecrawl"],
      }),
      llmRegistry: vi.fn().mockResolvedValue({ roles: {}, providers: [] }),
      enhancePrompt: vi.fn(),
      estimateRunCost: vi.fn(),
    },
  };
});

beforeEach(() => {
  vi.clearAllMocks();
  localStorageStore.clear();
  installLocalStorageMock();
  vi.mocked(api.llmRegistry).mockResolvedValue({ roles: {}, providers: [] } as never);
  vi.mocked(api.getSearchUsage).mockResolvedValue({
    usage_mode: "auto",
    fallback_chain: ["brave", "tavily", "exa", "firecrawl"],
  });
  vi.mocked(api.estimateRunCost).mockImplementation(() => new Promise(() => undefined));
  vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [{ id: "s1", title: "Quantum thread", updated_at: 1, latest_run_id: "run-1" }],
  });
  vi.mocked(api.searchHistory).mockResolvedValue({ results: [], scope: "intra" });
});

describe("SessionThread", () => {
  it("discovers a paused or interrupted latest run even before an assistant message exists", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({ messages: [] });
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [
        {
          id: "s1",
          title: "Paused Deep thread",
          updated_at: 2,
          latest_run_id: "paused-run",
          latest_run_status: "paused",
          resumable_checkpoint: {
            run_id: "paused-run",
            lineage_id: "lineage-1",
            revision: 1,
            schema_version: 1,
            stage: "plan_ready",
            disposition: "ready",
            integrity: "valid",
            resume_readiness: "ready",
            reason_codes: [],
            claimed_child_run_id: null,
          },
        },
      ],
    });

    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );

    expect(await screen.findByTestId("run-panel")).toHaveTextContent("paused-run");
  });

  it("renders prior turns with a link to the full run", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({
      messages: [
        { id: "m1", session_id: "s1", role: "user", content: "What is a qubit?", created_at: 1 },
        {
          id: "m2",
          session_id: "s1",
          role: "assistant",
          content: "A qubit is a quantum bit.",
          created_at: 2,
          metadata: JSON.stringify({ run_id: "run-1", mode: "fast" }),
        },
      ],
    });
    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );
    expect(await screen.findByText("What is a qubit?")).toBeInTheDocument();
    expect(screen.getByText("A qubit is a quantum bit.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /查看/ }));
    expect(await screen.findByTestId("run-panel")).toHaveTextContent("run-1");
  });

  it("submits a follow-up with the session_id and opens the new run in the split pane", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({
      messages: [
        { id: "m1", session_id: "s1", role: "user", content: "First question", created_at: 1 },
      ],
    });
    vi.mocked(api.createRun).mockResolvedValue({ run_id: "run-2", session_id: "s1", status: "queued" });
    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );
    await screen.findByText("First question");
    fireEvent.change(screen.getByLabelText("Research question"), {
      target: { value: "Follow-up question" },
    });
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    fireEvent.change(screen.getByLabelText("Search country"), { target: { value: "SG" } });
    fireEvent.change(screen.getByLabelText("Search language"), { target: { value: "zh-hans" } });
    fireEvent.change(screen.getByLabelText("搜索结果界面语言"), { target: { value: "zh-Hans-SG" } });
    fireEvent.click(screen.getByLabelText("Send"));
    await waitFor(() =>
      expect(api.createRun).toHaveBeenCalledWith(
        expect.objectContaining({
          query: "Follow-up question",
          session_id: "s1",
          mode: "fast",
          search_options: { country: "SG", search_lang: "zh-hans", ui_language: "zh-Hans-SG" },
        }),
      ),
    );
    expect(navigate).not.toHaveBeenCalledWith("/run/run-2");
    expect(await screen.findByTestId("run-panel")).toHaveTextContent("run-2");
  });

  it("persists a per-session split ratio after pointer drag", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({
      messages: [{ id: "m1", session_id: "s1", role: "user", content: "First question", created_at: 1 }],
    });
    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );
    await screen.findByText("First question");
    const layout = screen.getByTestId("session-split-layout");
    vi.spyOn(layout, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 1000,
      bottom: 600,
      width: 1000,
      height: 600,
      toJSON: () => ({}),
    });

    fireEvent.pointerDown(screen.getByTestId("session-split-handle"), { clientX: 480, pointerId: 1 });
    const moveEvent = new Event("pointermove") as PointerEvent;
    Object.defineProperty(moveEvent, "clientX", { value: 650 });
    act(() => {
      window.dispatchEvent(moveEvent);
      window.dispatchEvent(new Event("pointerup"));
    });

    await waitFor(() =>
      expect(window.localStorage.getItem("deepresearch-session-thread-split-ratio:s1")).toBe("0.65"),
    );
    expect(window.localStorage.getItem("deepresearch-session-thread-split-ratio")).toBeNull();
  });

  it("supports keyboard resizing with clamped separator values", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({
      messages: [{ id: "m1", session_id: "s1", role: "user", content: "First question", created_at: 1 }],
    });
    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );
    await screen.findByText("First question");
    const handle = screen.getByRole("separator", { name: /resize session split pane/i });

    fireEvent.keyDown(handle, { key: "ArrowRight" });
    await waitFor(() => expect(window.localStorage.getItem("deepresearch-session-thread-split-ratio:s1")).toBe("0.5"));

    fireEvent.keyDown(handle, { key: "End" });
    await waitFor(() => expect(handle).toHaveAttribute("aria-valuenow", "0.75"));

    fireEvent.keyDown(handle, { key: "Home" });
    await waitFor(() => expect(handle).toHaveAttribute("aria-valuenow", "0.25"));
  });

  it("initializes from mode_default and only saves defaults explicitly", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({
      messages: [
        {
          id: "m1",
          session_id: "s1",
          role: "assistant",
          content: "Older deep answer",
          created_at: 1,
          metadata: JSON.stringify({ run_id: "run-1", mode: "deep" }),
        },
      ],
    });
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [{ id: "s1", title: "Quantum thread", updated_at: 1, mode_default: "multiple" }],
    });
    vi.mocked(api.patchSession).mockResolvedValue({
      session_id: "s1",
      session: { id: "s1", title: "Quantum thread", updated_at: 1, mode_default: "council" },
    });
    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );

    expect(await screen.findByText("This is the default")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Models Council"));
    fireEvent.click(screen.getByRole("button", { name: /set as session default/i }));

    await waitFor(() => expect(api.patchSession).toHaveBeenCalledWith("s1", { mode_default: "council" }));
  });

  it("persists and clears the search strategy only for the current session", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({
      messages: [{ id: "m1", session_id: "s1", role: "user", content: "First question", created_at: 1 }],
    });
    vi.mocked(api.patchSession)
      .mockResolvedValueOnce({
        session_id: "s1",
        session: {
          id: "s1",
          title: "Quantum thread",
          updated_at: 1,
          search_usage_policy: {
            usage_mode: "auto",
            fallback_chain: ["brave", "tavily", "exa", "firecrawl"],
          },
        },
      })
      .mockResolvedValueOnce({
        session_id: "s1",
        session: { id: "s1", title: "Quantum thread", updated_at: 2, search_usage_policy: null },
      });

    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );
    await screen.findByText("First question");
    fireEvent.click(screen.getByLabelText("Options"));
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    fireEvent.click(screen.getByRole("button", { name: "Customize this session" }));

    await waitFor(() =>
      expect(api.patchSession).toHaveBeenCalledWith("s1", {
        search_usage_policy: {
          usage_mode: "auto",
          fallback_chain: ["brave", "tavily", "exa", "firecrawl"],
        },
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Use global default" }));
    await waitFor(() =>
      expect(api.patchSession).toHaveBeenLastCalledWith("s1", { search_usage_policy: null }),
    );
  });

  it("surfaces semantically related earlier turns as the user types (DR-ITER-053)", async () => {
    vi.mocked(api.listMessages).mockResolvedValue({
      messages: [
        { id: "m1", session_id: "s1", role: "user", content: "First question", created_at: 1 },
      ],
    });
    vi.mocked(api.searchHistory).mockResolvedValue({
      results: [
        {
          id: "old-1",
          session_id: "s1",
          role: "assistant",
          content: "Earlier we discussed quantum error correction codes.",
          created_at: 0,
          metadata: JSON.stringify({ run_id: "run-9", mode: "fast" }),
        },
      ],
      scope: "intra",
    });
    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );
    await screen.findByText("First question");
    fireEvent.change(screen.getByLabelText("Research question"), {
      target: { value: "error correction" },
    });
    await waitFor(() =>
      expect(api.searchHistory).toHaveBeenCalledWith(
        "s1",
        expect.objectContaining({ query: "error correction", scope: "intra" }),
      ),
    );
    expect(await screen.findByTestId("related-turns")).toBeInTheDocument();
    fireEvent.click(await screen.findByText("Earlier we discussed quantum error correction codes."));
    expect(await screen.findByTestId("run-panel")).toHaveTextContent("run-9");
  });

  it("shows a lock notice for an archived session", async () => {
    vi.mocked(api.listMessages).mockRejectedValue(new ApiError("SESSION_ARCHIVED", "archived"));
    render(
      <MemoryRouter>
        <SessionThread />
      </MemoryRouter>,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("已归档");
  });
});
