import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { History } from "./History";
import { api } from "@/lib/api";

const navigate = vi.fn();
vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>();
  return { ...actual, useNavigate: () => navigate };
});
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      listSessions: vi.fn(),
      patchSession: vi.fn(),
      archiveSession: vi.fn(),
      deleteSession: vi.fn(),
    },
  };
});

describe("History", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows mode badge for the latest run", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [
        { id: "s1", title: "Fast topic", updated_at: 1, latest_run_id: "run-9", latest_run_mode: "fast" },
      ],
    });
    render(
      <MemoryRouter>
        <History />
      </MemoryRouter>,
    );
    expect(await screen.findByLabelText("Fast Search")).toHaveTextContent("Fast");
  });

  it("marks a paused Deep run as resumable and opens its session", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [
        {
          id: "s1",
          title: "Paused research",
          updated_at: 1,
          latest_run_id: "run-paused",
          latest_run_mode: "deep",
          latest_run_status: "paused",
          resumable_checkpoint: {
            run_id: "run-paused",
            lineage_id: "lineage-1",
            revision: 2,
            schema_version: 1,
            stage: "final_evidence_ready",
            disposition: "ready",
            integrity: "valid",
            resume_readiness: "degraded",
            reason_codes: ["source_tombstoned"],
            claimed_child_run_id: null,
          },
        },
      ],
    });

    render(
      <MemoryRouter>
        <History />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Resume")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Paused research"));
    expect(navigate).toHaveBeenCalledWith("/session/s1");
  });

  it("navigates to the session thread", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [{ id: "s1", title: "First topic", updated_at: 1, latest_run_id: "run-9" }],
    });
    render(
      <MemoryRouter>
        <History />
      </MemoryRouter>,
    );
    await waitFor(() => expect(api.listSessions).toHaveBeenCalled());
    fireEvent.click(await screen.findByText("First topic"));
    expect(navigate).toHaveBeenCalledWith("/session/s1");
  });

  it("groups pinned sessions and unpins through session patch", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [
        { id: "s1", title: "Pinned topic", updated_at: 2, pinned: true, latest_run_id: "run-1" },
        { id: "s2", title: "Recent topic", updated_at: 1, pinned: false, latest_run_id: "run-2" },
      ],
    });
    vi.mocked(api.patchSession).mockResolvedValue({
      session_id: "s1",
      session: { id: "s1", title: "Pinned topic", updated_at: 2, pinned: false, latest_run_id: "run-1" },
    });
    render(
      <MemoryRouter>
        <History />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Pinned")).toBeInTheDocument();
    expect(screen.getByText("Recent")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Unpin Pinned topic/i }));

    await waitFor(() => expect(api.patchSession).toHaveBeenCalledWith("s1", { pinned: false }));
  });

  it("does not pin more than 10 sessions", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [
        ...Array.from({ length: 10 }, (_, index) => ({
          id: `p${index}`,
          title: `Pinned ${index}`,
          updated_at: index,
          pinned: true,
        })),
        { id: "s-new", title: "Overflow", updated_at: 20, pinned: false },
      ],
    });
    render(
      <MemoryRouter>
        <History />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByRole("button", { name: /Pin Overflow/i }));

    expect(api.patchSession).not.toHaveBeenCalled();
  });

  it("opens the thread even for sessions without a run", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [{ id: "s2", title: "Empty", updated_at: 1, latest_run_id: null }],
    });
    render(
      <MemoryRouter>
        <History />
      </MemoryRouter>,
    );
    fireEvent.click(await screen.findByText("Empty"));
    expect(navigate).toHaveBeenCalledWith("/session/s2");
  });

  it("archives a session after confirmation", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [{ id: "s3", title: "To archive", updated_at: 1, latest_run_id: "run-1" }],
    });
    vi.mocked(api.archiveSession).mockResolvedValue({ session_id: "s3", archived: true });
    render(
      <MemoryRouter>
        <History />
      </MemoryRouter>,
    );
    await waitFor(() => expect(api.listSessions).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: /Archive To archive/i }));
    fireEvent.click(await screen.findByRole("button", { name: /^Archive$/i }));
    await waitFor(() => expect(api.archiveSession).toHaveBeenCalledWith("s3"));
    expect(screen.queryByText("To archive")).toBeNull();
  });

  it("deletes a session after confirmation", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({
      sessions: [{ id: "s4", title: "To delete", updated_at: 1, latest_run_id: null }],
    });
    vi.mocked(api.deleteSession).mockResolvedValue({ session_id: "s4", deleted: true });
    render(
      <MemoryRouter>
        <History />
      </MemoryRouter>,
    );
    await waitFor(() => expect(api.listSessions).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: /Delete To delete/i }));
    expect(await screen.findByText(/workspace session summary/i)).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: /^Delete$/i }));
    await waitFor(() => expect(api.deleteSession).toHaveBeenCalledWith("s4"));
    expect(screen.queryByText("To delete")).toBeNull();
  });
});
