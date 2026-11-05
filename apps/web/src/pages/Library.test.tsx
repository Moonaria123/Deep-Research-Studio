import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Library } from "./Library";
import { api } from "@/lib/api";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      listDocuments: vi.fn(),
      listNotes: vi.fn(),
      addNote: vi.fn(),
      indexUrl: vi.fn(),
      listLibraryDocuments: vi.fn(),
      listLibraryNotes: vi.fn(),
      addLibraryNote: vi.fn(),
      uploadLibraryReport: vi.fn(),
      listLibraryReports: vi.fn(),
      getLibraryReport: vi.fn(),
      listSourceSessions: vi.fn(),
      listSourceSnapshots: vi.fn(),
      getLibrarySnapshot: vi.fn(),
      importLibraryDocument: vi.fn(),
      deleteLibraryDocument: vi.fn(),
    },
  };
});

function renderLibrary(initialPath = "/library") {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="/library/sources/:documentId" element={<Library />} />
        <Route path="/library/reports/:reportId" element={<Library />} />
        <Route path="/library/notes/:noteId" element={<Library />} />
        <Route path="/library/reports" element={<Library />} />
        <Route path="/library/notes" element={<Library />} />
        <Route path="/library" element={<Library />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Library", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.listLibraryDocuments).mockResolvedValue({ documents: [] });
    vi.mocked(api.listLibraryNotes).mockResolvedValue({ notes: [] });
    vi.mocked(api.listLibraryReports).mockResolvedValue({ reports: [] });
    vi.mocked(api.listSourceSessions).mockResolvedValue({ artifacts: [] });
    vi.mocked(api.listSourceSnapshots).mockResolvedValue({ snapshots: [] });
    vi.mocked(api.getLibrarySnapshot).mockResolvedValue({ snapshot: { id: "snap1", url: "", status: "ok", text: "" } });
    vi.mocked(api.addNote).mockResolvedValue({ note_id: "n1" });
    vi.mocked(api.addLibraryNote).mockResolvedValue({ note_id: "n1" });
    vi.mocked(api.uploadLibraryReport).mockResolvedValue({ report_id: "rep_upload" });
    vi.mocked(api.importLibraryDocument).mockResolvedValue({
      status: "created",
      document: {
        id: "local-doc",
        title: "Imported document",
        filename: "document.pdf",
        content_type: "pdf",
        origin: "local",
        artifact_type: "library_document",
        content_sha256: "a".repeat(64),
        chunk_count: 1,
      },
      warnings: [],
    });
    vi.mocked(api.deleteLibraryDocument).mockResolvedValue({
      status: "deleted",
      document_id: "local-doc",
      origin: "local",
      artifact_type: "library_document",
      chunks_deleted: 1,
      snapshots_deleted: 0,
      provenance_deleted: 1,
      warnings: [],
    });
  });

  it("loads and renders persisted documents on mount", async () => {
    vi.mocked(api.listLibraryDocuments).mockResolvedValue({
      documents: [
        { id: "d1", url: "https://e.com/a", title: "Doc A", content_type: "web", created_at: 1, chunks: 3 },
      ],
    });
    renderLibrary();
    await waitFor(() => expect(api.listLibraryDocuments).toHaveBeenCalled());
    expect(await screen.findByText("https://e.com/a")).toBeTruthy();
    expect(screen.getByText(/3 chunks/i)).toBeTruthy();
  });

  it("does not render legacy memory-derived local URLs in Sources", async () => {
    vi.mocked(api.listLibraryDocuments).mockResolvedValue({
      documents: [
        { id: "d1", url: "local://library/memory/m1", title: "Memory artifact", content_type: "memory", created_at: 1, chunks: 1 },
        { id: "d2", url: "https://e.com/a", title: "Doc A", content_type: "web", created_at: 2, chunks: 3 },
      ],
    });

    renderLibrary();

    await waitFor(() => expect(api.listLibraryDocuments).toHaveBeenCalled());
    expect(await screen.findByText("https://e.com/a")).toBeTruthy();
    expect(screen.queryByText("local://library/memory/m1")).not.toBeInTheDocument();
  });

  it("imports a local document, refreshes, and highlights an existing source", async () => {
    const imported = {
      id: "local-doc",
      url: null,
      title: "Imported document",
      filename: "document.pdf",
      content_type: "pdf",
      created_at: 1,
      chunks: 1,
      origin: "local" as const,
      artifact_type: "library_document" as const,
      artifact_id: "local-doc",
      deletable: true,
      content_sha256: "a".repeat(64),
    };
    vi.mocked(api.importLibraryDocument).mockResolvedValue({
      status: "existing",
      document: {
        id: imported.id,
        title: imported.title,
        filename: imported.filename,
        content_type: "pdf",
        origin: "local",
        artifact_type: "library_document",
        content_sha256: imported.content_sha256,
        chunk_count: 1,
      },
      warnings: [],
    });
    vi.mocked(api.listLibraryDocuments)
      .mockResolvedValueOnce({ documents: [] })
      .mockResolvedValue({ documents: [imported] });

    renderLibrary();
    const file = new File(["%PDF fixture"], "document.pdf", { type: "application/pdf" });
    fireEvent.change(await screen.findByLabelText("Local document file"), { target: { files: [file] } });
    fireEvent.change(screen.getByLabelText("Local document title"), { target: { value: "Imported document" } });
    fireEvent.click(screen.getByRole("button", { name: /^import$/i }));

    await waitFor(() =>
      expect(api.importLibraryDocument).toHaveBeenCalledWith(file, "Imported document"),
    );
    expect(await screen.findByText(/already imported/i)).toBeInTheDocument();
    expect(await screen.findByText(/extracted text is searchable/i)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /open latest web page/i })).not.toBeInTheDocument();
  });

  it("deletes a user-visible source only after confirmation and refreshes permanently", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const local = {
      id: "local-doc",
      url: null,
      title: "Imported document",
      filename: "document.pdf",
      content_type: "pdf",
      created_at: 1,
      chunks: 1,
      origin: "local" as const,
      artifact_type: "library_document" as const,
      artifact_id: "local-doc",
      deletable: true,
    };
    vi.mocked(api.listLibraryDocuments)
      .mockResolvedValueOnce({ documents: [local] })
      .mockResolvedValue({ documents: [] });

    renderLibrary();
    fireEvent.click(await screen.findByRole("button", { name: /delete imported document/i }));

    await waitFor(() => expect(api.deleteLibraryDocument).toHaveBeenCalledWith("local-doc"));
    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/historical citations and reports are preserved/i));
    expect(await screen.findByText(/historical citations were preserved/i)).toBeInTheDocument();
    expect(screen.queryByText("Imported document")).not.toBeInTheDocument();
    confirm.mockRestore();
  });

  it("does not expose delete controls for non-deletable sources", async () => {
    vi.mocked(api.listLibraryDocuments).mockResolvedValue({
      documents: [
        {
          id: "d1",
          url: "https://e.com/a",
          title: "Protected source",
          content_type: "web",
          created_at: 1,
          chunks: 1,
          origin: "external",
          artifact_type: "web_document",
          deletable: false,
        },
      ],
    });

    renderLibrary();
    expect(await screen.findByText("Protected source")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /delete protected source/i })).not.toBeInTheDocument();
  });

  it("adds a note and refreshes the list", async () => {
    renderLibrary();
    await waitFor(() => expect(api.listLibraryNotes).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /notes/i }));
    fireEvent.change(screen.getByLabelText("New note"), { target: { value: "Cite primary sources" } });
    fireEvent.click(screen.getByRole("button", { name: /add note/i }));

    await waitFor(() => expect(api.addLibraryNote).toHaveBeenCalledWith({ content: "Cite primary sources" }));
    await waitFor(() => expect(api.listLibraryNotes).toHaveBeenCalledTimes(2));
  });

  it("links run-backed notes to their source run", async () => {
    vi.mocked(api.listLibraryNotes).mockResolvedValue({
      notes: [
        {
          id: "n1",
          content: "Run note",
          created_at: 1,
          session_id: "s1",
          run_id: "run_1",
          metadata: { source: "post_run" },
        },
      ],
    });

    renderLibrary();
    fireEvent.click(await screen.findByRole("button", { name: /notes/i }));

    const link = await screen.findByRole("link", { name: /run_1/i });
    expect(link).toHaveAttribute("href", "/run/run_1");
  });

  it("shows source snapshots and linked sessions", async () => {
    vi.mocked(api.listLibraryDocuments).mockResolvedValue({
      documents: [
        { id: "d1", url: "https://e.com/a", title: "Doc A", content_type: "web", created_at: 1, chunks: 2 },
      ],
    });
    vi.mocked(api.listSourceSessions).mockResolvedValue({
      artifacts: [{ session_id: "s1", run_id: "r1", created_at: 1 }],
    });
    vi.mocked(api.listSourceSnapshots).mockResolvedValue({
      snapshots: [{ id: "snap1", url: "https://e.com/a", status: "ok", captured_at: 1, mime_type: "text/plain", size_bytes: 12 }],
    });
    vi.mocked(api.getLibrarySnapshot).mockResolvedValue({
      snapshot: { id: "snap1", url: "https://e.com/a", status: "ok", text: "snapshot preview" },
    });

    renderLibrary();
    fireEvent.click(await screen.findByText("Doc A"));

    expect(await screen.findByText(/snapshot preview/i)).toBeTruthy();
    expect(api.listSourceSessions).toHaveBeenCalledWith("d1");
    expect(api.getLibrarySnapshot).toHaveBeenCalledWith("snap1");
  });

  it("shows reports tab and previews a selected report", async () => {
    vi.mocked(api.listLibraryReports).mockResolvedValue({
      reports: [{ id: "r1", title: "Report", mode: "deep", created_at: 1, run_id: "run_1", children: [] }],
    });
    vi.mocked(api.getLibraryReport).mockResolvedValue({
      report: { id: "r1", title: "Report", mode: "deep", created_at: 1, run_id: "run_1", body_md: "# Report\n\nBody" },
    });

    renderLibrary();
    fireEvent.click(await screen.findByRole("button", { name: /reports/i }));
    fireEvent.click(await screen.findByText("Report"));

    expect(await screen.findByText("Body")).toBeTruthy();
    expect(screen.getByRole("link", { name: /run_1/i })).toHaveAttribute("href", "/run/run_1");
    expect(api.getLibraryReport).toHaveBeenCalledWith("r1");
  });

  it("uploads a markdown report and refreshes report rows", async () => {
    renderLibrary();
    fireEvent.click(await screen.findByRole("button", { name: /reports/i }));

    const file = new File(["# Uploaded\n\nBody"], "uploaded.md", { type: "text/markdown" });
    fireEvent.change(screen.getByLabelText("Report file"), { target: { files: [file] } });
    fireEvent.change(screen.getByLabelText("Report title"), { target: { value: "Uploaded report" } });
    fireEvent.click(screen.getByRole("button", { name: /upload/i }));

    await waitFor(() => expect(api.uploadLibraryReport).toHaveBeenCalledTimes(1));
    expect(api.uploadLibraryReport).toHaveBeenCalledWith(file, "Uploaded report");
    await waitFor(() => expect(api.listLibraryReports).toHaveBeenCalledTimes(2));
  });

  it("opens the notes tab from a knowledge-graph deep link", async () => {
    vi.mocked(api.listLibraryNotes).mockResolvedValue({
      notes: [{ id: "note-1", content: "Linked note body", created_at: 1 }],
    });

    renderLibrary("/library/notes/note-1");

    expect(await screen.findByText("Linked note body")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /notes/i })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows a friendly tombstone for a deleted source deep link", async () => {
    renderLibrary("/library/sources/deleted-document");

    expect(
      await screen.findByText(/historical source is no longer in the current index/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/saved citations and reports remain available/i)).toBeInTheDocument();
    expect(api.listSourceSessions).not.toHaveBeenCalled();
    expect(api.listSourceSnapshots).not.toHaveBeenCalled();
  });

  it("opens and previews a report from a knowledge-graph deep link", async () => {
    vi.mocked(api.listLibraryReports).mockResolvedValue({
      reports: [{ id: "r1", title: "Deep report", mode: "deep", created_at: 1, run_id: "run_1", children: [] }],
    });
    vi.mocked(api.getLibraryReport).mockResolvedValue({
      report: { id: "r1", title: "Deep report", mode: "deep", created_at: 1, run_id: "run_1", body_md: "# Deep\n\nPreview" },
    });

    renderLibrary("/library/reports/r1");

    expect(await screen.findByText("Preview")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reports/i })).toHaveAttribute("aria-pressed", "true");
    await waitFor(() => expect(api.getLibraryReport).toHaveBeenCalledWith("r1"));
  });
});
