import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SessionMemoryPanel } from "./SessionMemoryPanel";
import { api } from "@/lib/api";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      listSessionMemories: vi.fn(),
      archiveMemory: vi.fn(),
      restoreMemory: vi.fn(),
      deleteMemory: vi.fn(),
    },
  };
});

const response = {
  sessions: [
    {
      session_id: "sess-active",
      session_title: "Active research",
      session_updated_at: 10,
      session_archived_at: null,
      memories: [
        {
          id: "mem-active",
          layer: "session" as const,
          content: "Active session memory",
          salience: 1,
          effective_salience: 1,
          archived_at: null,
        },
      ],
    },
    {
      session_id: "sess-archived",
      session_title: "Archived research",
      session_updated_at: 8,
      session_archived_at: 20,
      memories: [
        {
          id: "mem-archived",
          layer: "session" as const,
          content: "Archived session memory",
          salience: 1,
          effective_salience: 1,
          archived_at: 21,
        },
      ],
    },
  ],
  total: 2,
  limit: 50,
  offset: 0,
  truncated: false,
};

describe("SessionMemoryPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.listSessionMemories).mockResolvedValue(response);
    vi.mocked(api.archiveMemory).mockResolvedValue({ memory_id: "mem-active", archived: true });
    vi.mocked(api.restoreMemory).mockResolvedValue({
      memory_id: "mem-archived",
      archived: false,
      library_index_warning: null,
    });
    vi.mocked(api.deleteMemory).mockResolvedValue({ memory_id: "mem-active", deleted: true });
  });

  it("loads grouped active and archived sessions without Recall", async () => {
    render(<SessionMemoryPanel />);

    expect(await screen.findByText("Active research")).toBeInTheDocument();
    expect(screen.getByText("Archived research")).toBeInTheDocument();
    expect(screen.getByText("Active session memory")).toBeInTheDocument();
    expect(screen.getByText("Archived session memory")).toBeInTheDocument();
    expect(api.listSessionMemories).toHaveBeenCalledWith({
      workspace_key: "default",
      query: "",
      include_archived_sessions: true,
      include_archived_memories: true,
      limit: 50,
      offset: 0,
    });
  });

  it("searches and exposes only archive, restore, and delete actions", async () => {
    render(<SessionMemoryPanel />);
    await screen.findByText("Active research");

    fireEvent.change(screen.getByLabelText("Search Session memories"), { target: { value: "quasar" } });
    fireEvent.submit(screen.getByRole("search"));
    await waitFor(() =>
      expect(api.listSessionMemories).toHaveBeenLastCalledWith(expect.objectContaining({ query: "quasar" })),
    );

    fireEvent.click(screen.getByLabelText("Archive Session memory mem-active"));
    await waitFor(() => expect(api.archiveMemory).toHaveBeenCalledWith("mem-active"));

    fireEvent.click(screen.getByLabelText("Restore Session memory mem-archived"));
    await waitFor(() => expect(api.restoreMemory).toHaveBeenCalledWith("mem-archived"));

    fireEvent.click(screen.getByLabelText("Delete Session memory mem-active"));
    fireEvent.click(await screen.findByRole("button", { name: "Delete Session memory" }));
    await waitFor(() => expect(api.deleteMemory).toHaveBeenCalledWith("mem-active"));

    expect(screen.queryByLabelText(/Pin Session memory/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/Edit Session memory/)).not.toBeInTheDocument();
  });
});
