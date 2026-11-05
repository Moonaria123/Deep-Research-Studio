import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ArchivedHistory } from "./ArchivedHistory";
import { api } from "@/lib/api";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      listArchivedSessions: vi.fn(),
      restoreSession: vi.fn(),
      deleteSession: vi.fn(),
    },
  };
});

describe("ArchivedHistory", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows mode badge for archived sessions", async () => {
    vi.mocked(api.listArchivedSessions).mockResolvedValue({
      sessions: [
        {
          id: "s1",
          title: "Deep topic",
          created_at: 1,
          updated_at: 2,
          archived_at: 3,
          latest_run_mode: "deep",
        },
      ],
    });
    render(
      <MemoryRouter>
        <ArchivedHistory />
      </MemoryRouter>,
    );
    expect(await screen.findByLabelText("Deep Research")).toHaveTextContent("Deep");
  });

  it("lists archived sessions without navigation links", async () => {
    vi.mocked(api.listArchivedSessions).mockResolvedValue({
      sessions: [
        {
          id: "s1",
          title: "Old topic",
          created_at: 1,
          updated_at: 2,
          archived_at: 3,
        },
      ],
    });
    render(
      <MemoryRouter>
        <ArchivedHistory />
      </MemoryRouter>,
    );
    await waitFor(() => expect(api.listArchivedSessions).toHaveBeenCalled());
    expect(await screen.findByText("Old topic")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Old topic/i })).toBeNull();
  });

  it("restores a session", async () => {
    vi.mocked(api.listArchivedSessions).mockResolvedValue({
      sessions: [
        {
          id: "s1",
          title: "Restore me",
          created_at: 1,
          updated_at: 2,
          archived_at: 3,
        },
      ],
    });
    vi.mocked(api.restoreSession).mockResolvedValue({ session_id: "s1", archived: false });
    render(
      <MemoryRouter>
        <ArchivedHistory />
      </MemoryRouter>,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Restore" }));
    await waitFor(() => expect(api.restoreSession).toHaveBeenCalledWith("s1"));
  });

  it("deletes an archived session after showing workspace summary cascade copy", async () => {
    vi.mocked(api.listArchivedSessions).mockResolvedValue({
      sessions: [
        {
          id: "s2",
          title: "Delete archived",
          created_at: 1,
          updated_at: 2,
          archived_at: 3,
        },
      ],
    });
    vi.mocked(api.deleteSession).mockResolvedValue({ session_id: "s2", deleted: true });
    render(
      <MemoryRouter>
        <ArchivedHistory />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByRole("button", { name: /delete delete archived/i }));
    expect(await screen.findByText(/workspace session summary/i)).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(api.deleteSession).toHaveBeenCalledWith("s2"));
  });
});
