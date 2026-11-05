import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Memory } from "./Memory";
import { api } from "@/lib/api";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      recallMemory: vi.fn(),
      getMemory: vi.fn(),
      addMemory: vi.fn(),
      listArchivedMemories: vi.fn(),
      listSessionMemories: vi.fn(),
      archiveMemory: vi.fn(),
      restoreMemory: vi.fn(),
      patchMemory: vi.fn(),
      deleteMemory: vi.fn(),
      previewMemoryOptimization: vi.fn(),
      applyMemoryOptimization: vi.fn(),
      llmRegistry: vi.fn(),
      getMemoryDreamStatus: vi.fn(),
      startMemoryDream: vi.fn(),
      cancelMemoryDream: vi.fn(),
      updateMemoryDreamSchedule: vi.fn(),
      getMemorySettings: vi.fn(),
      updateMemorySettings: vi.fn(),
      getMemoryWriteAudit: vi.fn(),
      batchMemoryExpiry: vi.fn(),
    },
  };
});

describe("Memory add", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.recallMemory).mockResolvedValue({ memories: [] });
    vi.mocked(api.getMemory).mockResolvedValue({
      memory: { id: "m1", layer: "workspace", content: "Focused memory", salience: 1, effective_salience: 1 },
    });
    vi.mocked(api.addMemory).mockResolvedValue({ memory_id: "m1" });
    vi.mocked(api.listArchivedMemories).mockResolvedValue({ memories: [] });
    vi.mocked(api.listSessionMemories).mockResolvedValue({
      sessions: [],
      total: 0,
      limit: 50,
      offset: 0,
      truncated: false,
    });
    vi.mocked(api.archiveMemory).mockResolvedValue({ memory_id: "m1", archived: true });
    vi.mocked(api.restoreMemory).mockResolvedValue({ memory_id: "m1", archived: false, library_index_warning: null });
    vi.mocked(api.patchMemory).mockResolvedValue({
      memory_id: "m1",
      memory: { id: "m1", layer: "workspace", content: "Updated memory", salience: 0.5, effective_salience: 0.5 },
      library_index_warning: null,
    });
    vi.mocked(api.deleteMemory).mockResolvedValue({ memory_id: "m1", deleted: true });
    vi.mocked(api.previewMemoryOptimization).mockResolvedValue({ groups: [], threshold: 0.82 });
    vi.mocked(api.applyMemoryOptimization).mockResolvedValue({
      strategy: "merge",
      memory_id: "m3",
      archived_ids: ["m1", "m2"],
      summary: "merged",
      library_index_warning: null,
    });
    vi.mocked(api.llmRegistry).mockResolvedValue({
      providers: [],
      roles: { default: "", embed: "", memory: "local:memory-model", council: [] },
    });
    vi.mocked(api.getMemoryDreamStatus).mockResolvedValue({
      status: "idle",
      dream_run_id: null,
      workspace_key: "default",
      user_key: "default",
      started_at: null,
      finished_at: null,
      memories_created: 0,
      error: null,
      schedule: "off",
      next_scheduled_at: null,
    });
    vi.mocked(api.startMemoryDream).mockResolvedValue({
      status: "completed",
      dream_run_id: "dream_1",
      workspace_key: "default",
      user_key: "default",
      started_at: 1,
      finished_at: 2,
      memories_created: 2,
      error: null,
      schedule: "off",
      next_scheduled_at: null,
    });
    vi.mocked(api.cancelMemoryDream).mockResolvedValue({
      status: "cancelled",
      dream_run_id: "dream_queued",
      workspace_key: "default",
      user_key: "default",
      started_at: 1,
      finished_at: 2,
      updated_at: 2,
      cancel_requested_at: 2,
      backend: "rq",
      backend_job_id: "memory-dream-dream_queued",
      memories_created: 0,
      error: null,
      schedule: "off",
      next_scheduled_at: null,
    });
    vi.mocked(api.updateMemoryDreamSchedule).mockResolvedValue({
      schedule: "daily",
      next_scheduled_at: 86402,
    });
    vi.mocked(api.getMemorySettings).mockResolvedValue({
      post_turn_memory_refresh_enabled: true,
      workspace_memory_refresh_strategy: "incremental",
      memory_thinking_enabled: false,
      format_probe: null,
    });
    vi.mocked(api.getMemoryWriteAudit).mockResolvedValue({
      status: "idle",
      source: "none",
      workspace_key: "default",
      user_key: "default",
      attempted: 0,
      succeeded: 0,
      skipped: 0,
      failed: 0,
      error_code: null,
      checked_at: null,
      provider_ref: "",
      stale: false,
      incomplete_reasons: [],
    });
    vi.mocked(api.batchMemoryExpiry)
      .mockResolvedValueOnce({
        namespace: "workspace",
        dry_run: true,
        would_update: 2,
        updated: 0,
        skipped_pinned: 1,
        skipped_archived: 1,
        failed: 0,
      })
      .mockResolvedValue({
        namespace: "workspace",
        dry_run: false,
        would_update: 2,
        updated: 2,
        skipped_pinned: 1,
        skipped_archived: 1,
        failed: 0,
      });
    vi.mocked(api.updateMemorySettings)
      .mockResolvedValueOnce({
        post_turn_memory_refresh_enabled: false,
        workspace_memory_refresh_strategy: "incremental",
        memory_thinking_enabled: false,
        format_probe: null,
      })
      .mockResolvedValue({
        post_turn_memory_refresh_enabled: false,
        workspace_memory_refresh_strategy: "full",
        memory_thinking_enabled: false,
        format_probe: null,
      });
  });

  it("adds a memory to the selected layer and refreshes the list", async () => {
    render(<Memory />);
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText("New memory content"), {
      target: { value: "Prefer local-first tools" },
    });
    fireEvent.change(screen.getByLabelText("Memory layer"), { target: { value: "user" } });
    fireEvent.click(screen.getByRole("button", { name: /add memory/i }));

    await waitFor(() => expect(api.addMemory).toHaveBeenCalledTimes(1));
    expect(api.addMemory).toHaveBeenCalledWith({
      layer: "user",
      scope_key: "default",
      content: "Prefer local-first tools",
    });
    // List reloads after a successful add.
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(2));
    // Draft is cleared.
    expect((screen.getByLabelText("New memory content") as HTMLTextAreaElement).value).toBe("");
  });

  it("does not submit an empty memory", async () => {
    render(<Memory />);
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: /add memory/i }));
    expect(api.addMemory).not.toHaveBeenCalled();
  });

  it("refreshes the research profile and reloads memories", async () => {
    render(<Memory />);
    await waitFor(() => expect(api.getMemoryDreamStatus).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /refresh/i }));

    await waitFor(() => expect(api.startMemoryDream).toHaveBeenCalledTimes(1));
    expect(api.startMemoryDream).toHaveBeenCalledWith({
      workspace_key: "default",
      user_key: "default",
    });
    expect(await screen.findByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(2));
  });

  it("updates the dreaming schedule from the profile card", async () => {
    render(<Memory />);
    await waitFor(() => expect(api.getMemoryDreamStatus).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText("Dreaming schedule"), { target: { value: "daily" } });

    await waitFor(() => expect(api.updateMemoryDreamSchedule).toHaveBeenCalledWith({ schedule: "daily" }));
    expect(await screen.findByText(/next schedule/i)).toBeInTheDocument();
  });

  it("shows queued Dream status and cancels the durable job", async () => {
    vi.mocked(api.getMemoryDreamStatus).mockResolvedValueOnce({
      status: "queued",
      dream_run_id: "dream_queued",
      workspace_key: "default",
      user_key: "default",
      started_at: 1,
      finished_at: null,
      updated_at: 1,
      cancel_requested_at: null,
      backend: "rq",
      backend_job_id: "memory-dream-dream_queued",
      memories_created: 0,
      error: null,
      schedule: "off",
      next_scheduled_at: null,
    });

    render(<Memory />);

    expect(await screen.findByText("Queued")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(api.cancelMemoryDream).toHaveBeenCalledWith("dream_queued"));
    expect(await screen.findByText("Cancelled")).toBeInTheDocument();
  });

  it("persists research profile memory settings", async () => {
    render(<Memory />);
    await waitFor(() => expect(api.getMemorySettings).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("switch", { name: /post-turn memory refresh/i }));
    await waitFor(() =>
      expect(api.updateMemorySettings).toHaveBeenCalledWith({ post_turn_memory_refresh_enabled: false }),
    );

    fireEvent.change(screen.getByLabelText("Workspace memory refresh strategy"), { target: { value: "full" } });
    await waitFor(() =>
      expect(api.updateMemorySettings).toHaveBeenCalledWith({ workspace_memory_refresh_strategy: "full" }),
    );
  });

  it("shows write-audit counts and warnings", async () => {
    vi.mocked(api.getMemoryWriteAudit).mockResolvedValueOnce({
      status: "completed",
      source: "post_turn_memory_v1",
      workspace_key: "default",
      user_key: "default",
      attempted: 3,
      succeeded: 2,
      skipped: 1,
      failed: 1,
      error_code: "JSON_PARSE_FAILED",
      checked_at: 2,
      provider_ref: "openai:gpt-4o-mini",
      stale: false,
      incomplete_reasons: ["json_parse_failed"],
    });

    render(<Memory />);

    expect(await screen.findByText("2/3 succeeded")).toBeInTheDocument();
    expect(screen.getByText(/1 write item failed/i)).toBeInTheDocument();
    expect(screen.getByText(/JSON_PARSE_FAILED/i)).toBeInTheDocument();
  });

  it("shows staged research profile refresh warnings", async () => {
    vi.mocked(api.getMemoryDreamStatus).mockResolvedValueOnce({
      status: "completed",
      dream_run_id: "dream_warn",
      workspace_key: "default",
      user_key: "default",
      started_at: 1,
      finished_at: 2,
      memories_created: 1,
      error: null,
      error_code: "PARTIAL_FAILURE",
      failed_stage: "session_distill",
      failed_session_id: "sess-bad-json",
      attempted_sessions: 2,
      successful_sessions: 1,
      failed_sessions: 1,
      user_profile_failed: true,
      failure_count: 2,
      incomplete: true,
      incomplete_reasons: ["session_failures"],
      structured_strategy: "json_schema_strict",
      response_shape: "prose_wrapped_json",
      validation_error: "JSON_PARSE_FAILED",
      schedule: "off",
      next_scheduled_at: null,
    });

    render(<Memory />);

    expect(await screen.findByText(/2 refresh items skipped/i)).toBeInTheDocument();
    expect(screen.getByText(/Code: PARTIAL_FAILURE/i)).toBeInTheDocument();
    expect(screen.getByText(/Stage: Session summary/i)).toBeInTheDocument();
    expect(screen.getByText(/Session: sess-bad-json/i)).toBeInTheDocument();
    expect(screen.getByText(/Sessions: 1\/2 succeeded/i)).toBeInTheDocument();
    expect(screen.getByText(/User profile skipped/i)).toBeInTheDocument();
    expect(screen.getByText(/Validation: JSON_PARSE_FAILED/i)).toBeInTheDocument();
  });

  it("archives an active workspace memory", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "workspace", content: "Keep local", salience: 1, effective_salience: 1 }],
    });
    render(<Memory />);
    await screen.findByText("Keep local");

    fireEvent.click(screen.getByLabelText("Archive memory m1"));

    await waitFor(() => expect(api.archiveMemory).toHaveBeenCalledWith("m1"));
    await waitFor(() => expect(api.listArchivedMemories).toHaveBeenCalled());
  });

  it("edits an active workspace memory and refreshes the list", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "workspace", content: "Old memory", salience: 0.7, effective_salience: 0.7 }],
    });
    render(<Memory />);
    await screen.findByText("Old memory");

    fireEvent.click(screen.getByLabelText("Edit memory m1"));
    fireEvent.change(await screen.findByLabelText("Edit memory content"), { target: { value: "Updated memory" } });
    fireEvent.change(screen.getByLabelText("Memory salience"), { target: { value: "0.4" } });
    fireEvent.click(screen.getByRole("button", { name: /save memory/i }));

    await waitFor(() =>
      expect(api.patchMemory).toHaveBeenCalledWith("m1", {
        content: "Updated memory",
        salience: 0.4,
      }),
    );
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(2));
    expect(screen.queryByText(/Library indexing failed/i)).not.toBeInTheDocument();
  });

  it("pins and unpins workspace memories", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "workspace", content: "Pin me", salience: 1, effective_salience: 1, pinned: false }],
    });
    vi.mocked(api.patchMemory).mockResolvedValueOnce({
      memory_id: "m1",
      memory: { id: "m1", layer: "workspace", content: "Pin me", salience: 1, effective_salience: 1, pinned: true },
      library_index_warning: null,
    });
    render(<Memory />);
    await screen.findByText("Pin me");

    fireEvent.click(screen.getByLabelText("Pin memory m1"));

    await waitFor(() => expect(api.patchMemory).toHaveBeenCalledWith("m1", { pinned: true }));
    expect(await screen.findByText("Pinned")).toBeInTheDocument();
  });

  it("disables TTL and archive controls while a memory is pinned", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "workspace", content: "Protected", salience: 1, effective_salience: 1, pinned: true }],
    });
    render(<Memory />);
    await screen.findByText("Protected");

    expect(screen.getByLabelText("Memory expiry preset m1")).toBeDisabled();
    expect(screen.getByLabelText("Custom expiry m1")).toBeDisabled();
    expect(screen.getByLabelText("Archive memory m1")).toBeDisabled();
  });

  it("loads and highlights an archived memory from the citation deep link", async () => {
    vi.mocked(api.getMemory).mockResolvedValueOnce({
      memory: {
        id: "archived-1",
        layer: "workspace",
        content: "Archived cited evidence",
        salience: 1,
        effective_salience: 1,
        archived_at: 123,
      },
    });

    render(
      <MemoryRouter initialEntries={["/memory/archived-1"]}>
        <Routes>
          <Route path="/memory/:memoryId" element={<Memory />} />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => expect(api.getMemory).toHaveBeenCalledWith("archived-1"));
    expect(await screen.findByText("Archived cited evidence")).toBeInTheDocument();
    expect(screen.getByText("Archived cited evidence").closest("li")).toHaveAttribute("data-memory-focused", "true");
  });

  it("updates memory TTL presets and custom date", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "user", content: "TTL memory", salience: 1, effective_salience: 1 }],
    });
    vi.mocked(api.patchMemory)
      .mockResolvedValueOnce({
        memory_id: "m1",
        memory: {
          id: "m1",
          layer: "user",
          content: "TTL memory",
          salience: 1,
          effective_salience: 1,
          expires_at: 999,
        },
        library_index_warning: null,
      })
      .mockResolvedValueOnce({
        memory_id: "m1",
        memory: {
          id: "m1",
          layer: "user",
          content: "TTL memory",
          salience: 1,
          effective_salience: 1,
          expires_at: 1893455999,
        },
        library_index_warning: null,
      });
    render(<Memory />);
    await screen.findByText("TTL memory");

    fireEvent.change(screen.getByLabelText("Memory expiry preset m1"), { target: { value: "30" } });
    await waitFor(() => expect(api.patchMemory).toHaveBeenCalledWith("m1", { expires_at: expect.any(Number) }));

    fireEvent.change(screen.getByLabelText("Custom expiry m1"), { target: { value: "2030-01-01" } });
    await waitFor(() => expect(api.patchMemory).toHaveBeenLastCalledWith("m1", { expires_at: expect.any(Number) }));
  });

  it("previews and applies workspace batch expiry from the research profile", async () => {
    render(<Memory />);
    await waitFor(() => expect(api.getMemorySettings).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText("Workspace memory expiry preset"), { target: { value: "30" } });
    fireEvent.click(screen.getByRole("button", { name: /apply workspace/i }));

    await waitFor(() =>
      expect(api.batchMemoryExpiry).toHaveBeenCalledWith({
        workspace_key: "default",
        user_key: "default",
        namespace: "workspace",
        expires_at: expect.any(Number),
        dry_run: true,
      }),
    );
    expect(
      await screen.findByRole("heading", { name: /apply memory expiry/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/will update/i)).toBeInTheDocument();
    expect(screen.getByText(/pinned skipped/i)).toBeInTheDocument();
    expect(screen.getByText(/archived skipped/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /apply expiry/i }));

    await waitFor(() =>
      expect(api.batchMemoryExpiry).toHaveBeenLastCalledWith({
        workspace_key: "default",
        user_key: "default",
        namespace: "workspace",
        expires_at: expect.any(Number),
        dry_run: false,
      }),
    );
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.listArchivedMemories).toHaveBeenCalledTimes(2));
  });

  it("uses local end-of-day timestamps for custom user batch expiry", async () => {
    vi.mocked(api.batchMemoryExpiry).mockReset();
    vi.mocked(api.batchMemoryExpiry).mockResolvedValueOnce({
      namespace: "user",
      dry_run: true,
      would_update: 1,
      updated: 0,
      skipped_pinned: 0,
      skipped_archived: 0,
      failed: 0,
    });
    render(<Memory />);
    await waitFor(() => expect(api.getMemorySettings).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText("User memory expiry preset"), { target: { value: "custom" } });
    fireEvent.change(screen.getByLabelText("User memory expiry custom date"), { target: { value: "2030-01-01" } });
    fireEvent.click(screen.getByRole("button", { name: /apply user/i }));

    await waitFor(() =>
      expect(api.batchMemoryExpiry).toHaveBeenCalledWith({
        workspace_key: "default",
        user_key: "default",
        namespace: "user",
        expires_at: Math.floor(new Date("2030-01-01T23:59:59").getTime() / 1000),
        dry_run: true,
      }),
    );
  });

  it("shows and clears Memory edit Library indexing warnings", async () => {
    vi.mocked(api.recallMemory).mockResolvedValue({
      memories: [{ id: "m1", layer: "user", content: "Needs edit", salience: 1, effective_salience: 1 }],
    });
    vi.mocked(api.patchMemory)
      .mockResolvedValueOnce({
        memory_id: "m1",
        memory: { id: "m1", layer: "user", content: "Needs edit", salience: 1, effective_salience: 1 },
        library_index_warning: {
          code: "library_index_failed",
          message: "Memory was updated, but Library re-indexing failed.",
        },
      })
      .mockResolvedValueOnce({
        memory_id: "m1",
        memory: { id: "m1", layer: "user", content: "Needs edit", salience: 1, effective_salience: 1 },
        library_index_warning: null,
      });
    render(<Memory />);
    await screen.findByText("Needs edit");

    fireEvent.click(screen.getByLabelText("Edit memory m1"));
    fireEvent.click(await screen.findByRole("button", { name: /save memory/i }));

    expect(await screen.findByText("Memory was updated, but Library re-indexing failed.")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Edit memory m1"));
    fireEvent.click(await screen.findByRole("button", { name: /save memory/i }));

    await waitFor(() =>
      expect(screen.queryByText("Memory was updated, but Library re-indexing failed.")).not.toBeInTheDocument(),
    );
  });

  it("does not render Edit for session or archived memories", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [{ id: "s1", layer: "session", content: "Session scoped", salience: 1, effective_salience: 1 }],
    });
    vi.mocked(api.listArchivedMemories).mockResolvedValueOnce({
      memories: [{ id: "a1", layer: "workspace", content: "Archived scoped", salience: 1, effective_salience: 1, archived_at: 1 }],
    });
    render(<Memory />);
    await screen.findByText("Session scoped");

    expect(screen.queryByLabelText("Edit memory s1")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /archived/i }));
    await screen.findByText("Archived scoped");
    expect(screen.queryByLabelText("Edit memory a1")).not.toBeInTheDocument();
  });

  it("restores an archived memory from the archived dialog", async () => {
    vi.mocked(api.listArchivedMemories).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "user", content: "Archived item", salience: 1, effective_salience: 1, archived_at: 1 }],
    });
    render(<Memory />);
    await waitFor(() => expect(api.listArchivedMemories).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /archived/i }));
    fireEvent.click(await screen.findByLabelText("Restore memory m1"));

    await waitFor(() => expect(api.restoreMemory).toHaveBeenCalledWith("m1"));
    expect(screen.queryByText(/Library re-indexing failed/i)).not.toBeInTheDocument();
  });

  it("shows a non-blocking restore warning when Library re-indexing degrades", async () => {
    vi.mocked(api.listArchivedMemories).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "user", content: "Archived warning", salience: 1, effective_salience: 1, archived_at: 1 }],
    });
    vi.mocked(api.restoreMemory).mockResolvedValueOnce({
      memory_id: "m1",
      archived: false,
      library_index_warning: {
        code: "library_index_failed",
        message: "Memory was restored, but Library re-indexing failed.",
      },
    });
    render(<Memory />);
    await waitFor(() => expect(api.listArchivedMemories).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /archived/i }));
    fireEvent.click(await screen.findByLabelText("Restore memory m1"));

    expect(await screen.findByText("Memory was restored, but Library re-indexing failed.")).toBeInTheDocument();
  });

  it("requires confirmation before deleting a memory", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "workspace", content: "Delete me", salience: 1, effective_salience: 1 }],
    });
    render(<Memory />);
    await screen.findByText("Delete me");

    fireEvent.click(screen.getByLabelText("Delete memory m1"));
    fireEvent.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(api.deleteMemory).toHaveBeenCalledWith("m1"));
  });

  it("deletes an archived memory from the archived dialog and refreshes archived memories", async () => {
    vi.mocked(api.listArchivedMemories).mockResolvedValueOnce({
      memories: [
        {
          id: "m-arch",
          layer: "workspace",
          content: "Archived delete target",
          salience: 1,
          effective_salience: 1,
          archived_at: 1,
        },
      ],
    });
    render(<Memory />);
    await waitFor(() => expect(api.listArchivedMemories).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /archived/i }));
    expect(await screen.findByText(/archived 1\/1\/1970|1970/i)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Delete archived memory m-arch"));
    fireEvent.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(api.deleteMemory).toHaveBeenCalledWith("m-arch"));
    await waitFor(() => expect(api.listArchivedMemories).toHaveBeenCalledTimes(2));
  });

  it("disables archive while the operation is busy to avoid duplicate submits", async () => {
    let resolveArchive: (value: { memory_id: string; archived: boolean }) => void = () => undefined;
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [{ id: "m1", layer: "workspace", content: "Busy archive", salience: 1, effective_salience: 1 }],
    });
    vi.mocked(api.archiveMemory).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveArchive = resolve;
      }),
    );
    render(<Memory />);
    await screen.findByText("Busy archive");

    const archiveButton = screen.getByLabelText("Archive memory m1");
    fireEvent.click(archiveButton);
    fireEvent.click(archiveButton);

    expect(api.archiveMemory).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(archiveButton).toBeDisabled());
    resolveArchive({ memory_id: "m1", archived: true });
    await waitFor(() => expect(api.listArchivedMemories).toHaveBeenCalled());
  });

  it("does not render the unmanaged Session filter chip", async () => {
    render(<Memory />);
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(1));

    expect(screen.queryByRole("button", { name: /^session$/i })).not.toBeInTheDocument();
  });

  it("renders workspace session summaries with their session label", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [
        {
          id: "mw1",
          layer: "workspace",
          content: "Session summary: focus on local Library boundaries.",
          salience: 2.5,
          effective_salience: 2.5,
          metadata: {
            source: "session_distill_v1",
            session_id: "sess-1",
            session_title: "Library boundary review",
          },
        },
      ],
    });

    render(<Memory />);

    expect(await screen.findByText("Session summary · Library boundary review")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open source session/i })).toHaveAttribute("href", "/session/sess-1");
    expect(screen.getByText("Session summary: focus on local Library boundaries.")).toBeInTheDocument();
  });

  it("does not render a source session link without session metadata", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [
        {
          id: "mw-no-session",
          layer: "workspace",
          content: "Session summary: no source session id.",
          salience: 2,
          effective_salience: 2,
          metadata: { source: "session_distill_v1" },
        },
      ],
    });

    render(<Memory />);

    expect(await screen.findByText("Session summary · Untitled session")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /open source session/i })).not.toBeInTheDocument();
  });

  it("renders user profile sections from structured metadata", async () => {
    vi.mocked(api.recallMemory).mockResolvedValueOnce({
      memories: [
        {
          id: "mu1",
          layer: "user",
          content: "## Research Preferences\nPrefers concise evidence-backed answers.",
          salience: 3,
          effective_salience: 3,
          metadata: {
            source: "user_profile_v1",
            sections: [
              { id: "preferences", title: "Research Preferences", body: "Prefers concise evidence-backed answers." },
              { id: "constraints", title: "Constraints", body: "Avoid non-local storage." },
            ],
          },
        },
      ],
    });

    render(<Memory />);

    expect(await screen.findByText("User profile")).toBeInTheDocument();
    expect(screen.getByText("Research Preferences")).toBeInTheDocument();
    expect(screen.getByText("Prefers concise evidence-backed answers.")).toBeInTheDocument();
    expect(screen.getByText("Avoid non-local storage.")).toBeInTheDocument();
  });

  it("previews and applies duplicate memory merge", async () => {
    vi.mocked(api.previewMemoryOptimization).mockResolvedValue({
      threshold: 0.82,
      groups: [
        {
          group_id: "g1",
          similarity: 0.9,
          suggested_summary: "Prefer local-first answers",
          items: [
            { id: "m1", layer: "workspace", content: "Prefer local-first answers", salience: 1, effective_salience: 1 },
            { id: "m2", layer: "workspace", content: "Prefer local-first responses", salience: 1, effective_salience: 1 },
          ],
          suggestions: [{ type: "merge", memory_ids: ["m1", "m2"] }],
        },
      ],
    });
    render(<Memory />);
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /review duplicates/i }));
    expect((await screen.findAllByText("Prefer local-first answers")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: /^merge$/i }));

    await waitFor(() =>
      expect(api.applyMemoryOptimization).toHaveBeenCalledWith({
        strategy: "merge",
        memory_ids: ["m1", "m2"],
      }),
    );
    expect(screen.queryByText(/Library indexing failed/i)).not.toBeInTheDocument();
  });

  it("shows a merge Library indexing warning without blocking the optimizer action", async () => {
    vi.mocked(api.previewMemoryOptimization).mockResolvedValue({
      threshold: 0.82,
      groups: [
        {
          group_id: "g1",
          similarity: 0.9,
          suggested_summary: "Prefer local-first answers",
          items: [
            { id: "m1", layer: "workspace", content: "Prefer local-first answers", salience: 1, effective_salience: 1 },
            { id: "m2", layer: "workspace", content: "Prefer local-first responses", salience: 1, effective_salience: 1 },
          ],
          suggestions: [{ type: "merge", memory_ids: ["m1", "m2"] }],
        },
      ],
    });
    vi.mocked(api.applyMemoryOptimization).mockResolvedValueOnce({
      strategy: "merge",
      memory_id: "m3",
      archived_ids: ["m1", "m2"],
      summary: "merged",
      library_index_warning: {
        code: "library_index_failed",
        message: "Merged memory was saved, but Library indexing failed.",
      },
    });
    render(<Memory />);
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /review duplicates/i }));
    fireEvent.click(await screen.findByRole("button", { name: /^merge$/i }));

    expect(await screen.findByText("Merged memory was saved, but Library indexing failed.")).toBeInTheDocument();
  });

  it("renders archive and downrank optimizer suggestions and applies their payloads", async () => {
    vi.mocked(api.previewMemoryOptimization).mockResolvedValue({
      threshold: 0.82,
      groups: [
        {
          group_id: "g1",
          similarity: 0.9,
          suggested_summary: "Prefer local-first answers",
          items: [
            { id: "m1", layer: "workspace", content: "Prefer local-first answers", salience: 2, effective_salience: 2 },
            { id: "m2", layer: "workspace", content: "Prefer local-first responses", salience: 1, effective_salience: 1 },
          ],
          suggestions: [
            { type: "archive", memory_ids: ["m2"], keep_id: "m1" },
            { type: "downrank", memory_ids: ["m2"], factor: 0.25 },
          ],
        },
      ],
    });
    render(<Memory />);
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /review duplicates/i }));
    fireEvent.click(await screen.findByRole("button", { name: /archive duplicates/i }));

    await waitFor(() =>
      expect(api.applyMemoryOptimization).toHaveBeenCalledWith({
        strategy: "archive",
        memory_ids: ["m2"],
      }),
    );

    vi.mocked(api.applyMemoryOptimization).mockClear();
    fireEvent.click(await screen.findByRole("button", { name: /lower priority/i }));

    await waitFor(() =>
      expect(api.applyMemoryOptimization).toHaveBeenCalledWith({
        strategy: "downrank",
        memory_ids: ["m2"],
        factor: 0.25,
      }),
    );
  });

  it("disables merge without a memory role but leaves local optimizer actions available", async () => {
    vi.mocked(api.llmRegistry).mockResolvedValueOnce({
      providers: [],
      roles: { default: "", embed: "", memory: "", council: [] },
    });
    vi.mocked(api.previewMemoryOptimization).mockResolvedValueOnce({
      threshold: 0.82,
      groups: [
        {
          group_id: "g1",
          similarity: 0.9,
          suggested_summary: "Prefer local-first answers",
          items: [
            { id: "m1", layer: "workspace", content: "Prefer local-first answers", salience: 2, effective_salience: 2 },
            { id: "m2", layer: "workspace", content: "Prefer local-first responses", salience: 1, effective_salience: 1 },
          ],
          suggestions: [
            { type: "merge", memory_ids: ["m1", "m2"] },
            { type: "archive", memory_ids: ["m2"], keep_id: "m1" },
            { type: "downrank", memory_ids: ["m2"], factor: 0.5 },
          ],
        },
      ],
    });
    render(<Memory />);
    await waitFor(() => expect(api.recallMemory).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /review duplicates/i }));

    expect(await screen.findByRole("button", { name: /^merge$/i })).toBeDisabled();
    expect(screen.getByRole("link", { name: /set memory processing model/i })).toHaveAttribute("href", "/settings");
    expect(screen.getByRole("button", { name: /archive duplicates/i })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: /lower priority/i })).not.toBeDisabled();
  });
});
