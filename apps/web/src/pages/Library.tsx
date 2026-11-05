import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import {
  ExternalLink,
  FileText,
  Globe,
  History,
  Loader2,
  Plus,
  StickyNote,
  Trash2,
  Upload,
} from "lucide-react";
import { AnswerBody } from "@/components/AnswerBody";
import { api, ApiError } from "@/lib/api";
import { libraryPathForTab, libraryTabFromPath } from "@/lib/libraryRoutes";
import type { LibraryDocument, LibraryProvenance, LibraryReport, Note, PageSnapshot } from "@/lib/types";
import { cn, domainOf, formatTime } from "@/lib/utils";
import { useGsapEntrance } from "@/hooks/useGsapEntrance";
import { IS_DEMO } from "@/lib/runtime";
import { Card, CardContent } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

type Tab = "sources" | "notes" | "reports";

const TABS: { id: Tab; label: string }[] = [
  { id: "sources", label: "Sources" },
  { id: "notes", label: "Notes" },
  { id: "reports", label: "Reports" },
];

function isExternalSource(doc: LibraryDocument): boolean {
  const url = String(doc.url || "").toLowerCase();
  return url.startsWith("http://") || url.startsWith("https://");
}

function isLocalDocument(doc: LibraryDocument): boolean {
  return doc.origin === "local" && doc.artifact_type === "library_document";
}

function isVisibleSource(doc: LibraryDocument): boolean {
  return isExternalSource(doc) || isLocalDocument(doc);
}

function noteTag(note: Note): string {
  return note.metadata?.source === "post_run" ? "Run note" : "Manual";
}

function artifactHref(item: { run_id?: string | null; session_id?: string | null }): string {
  return item.run_id ? `/run/${item.run_id}` : item.session_id ? `/session/${item.session_id}` : "";
}

export function Library() {
  const navigate = useNavigate();
  const location = useLocation();
  const { noteId, reportId, documentId } = useParams<{
    noteId?: string;
    reportId?: string;
    documentId?: string;
  }>();
  const [tab, setTab] = useState<Tab>(() => libraryTabFromPath(location.pathname));
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [docs, setDocs] = useState<LibraryDocument[]>([]);
  const [docsLoaded, setDocsLoaded] = useState(false);
  const [notes, setNotes] = useState<Note[]>([]);
  const [reports, setReports] = useState<LibraryReport[]>([]);
  const [draft, setDraft] = useState("");
  const [addingNote, setAddingNote] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadTitle, setUploadTitle] = useState("");
  const [uploadingReport, setUploadingReport] = useState(false);
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [documentTitle, setDocumentTitle] = useState("");
  const [importingDocument, setImportingDocument] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState("");
  const [sourceNotice, setSourceNotice] = useState("");
  const [selectedDoc, setSelectedDoc] = useState<LibraryDocument>();
  const [sourceSessions, setSourceSessions] = useState<LibraryProvenance[]>([]);
  const [sourceSnapshots, setSourceSnapshots] = useState<PageSnapshot[]>([]);
  const [snapshotPreview, setSnapshotPreview] = useState<PageSnapshot>();
  const [selectedReport, setSelectedReport] = useState<LibraryReport>();
  const documentInputRef = useRef<HTMLInputElement>(null);
  const deepLinkApplied = useRef({ note: "", report: "", source: "" });
  const ref = useGsapEntrance<HTMLDivElement>([tab, docs.length, notes.length, reports.length]);

  const fail = (e: unknown) => setError(e instanceof ApiError ? e.message : (e as Error).message);

  const loadDocs = async () => {
    setDocsLoaded(false);
    try {
      const res = await api.listLibraryDocuments();
      const visibleDocuments = res.documents.filter(isVisibleSource);
      setDocs(visibleDocuments);
      setDocsLoaded(true);
      return visibleDocuments;
    } catch (e) {
      fail(e);
      return [];
    }
  };

  const loadNotes = async () => {
    try {
      const res = await api.listLibraryNotes();
      setNotes(res.notes);
    } catch (e) {
      fail(e);
    }
  };

  const loadReports = async () => {
    try {
      const res = await api.listLibraryReports();
      setReports(res.reports);
    } catch (e) {
      fail(e);
    }
  };

  useEffect(() => {
    void loadDocs();
    void loadNotes();
    void loadReports();
  }, []);

  useEffect(() => {
    setTab(libraryTabFromPath(location.pathname));
  }, [location.pathname]);

  const switchTab = (nextTab: Tab) => {
    setTab(nextTab);
    navigate(libraryPathForTab(nextTab));
  };

  const addUrl = async () => {
    const target = url.trim();
    if (!target) return;
    setBusy(true);
    setError(undefined);
    try {
      await api.indexUrl(target);
      setUrl("");
      await loadDocs();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  const importDocument = async () => {
    if (!documentFile || importingDocument) return;
    if (documentFile.size > 10 * 1024 * 1024) {
      setError("Document exceeds the 10 MiB import limit.");
      return;
    }
    setImportingDocument(true);
    setError(undefined);
    setSourceNotice("");
    try {
      const result = await api.importLibraryDocument(documentFile, documentTitle.trim());
      const refreshed = await loadDocs();
      const imported = refreshed.find((item) => item.id === result.document.id);
      setDocumentFile(null);
      if (documentInputRef.current) documentInputRef.current.value = "";
      setDocumentTitle("");
      setSourceNotice(
        result.status === "existing"
          ? "This file was already imported. The existing source is highlighted."
          : "Local document imported without external provider calls.",
      );
      if (imported) {
        await selectSource(imported);
      }
    } catch (e) {
      fail(e);
    } finally {
      setImportingDocument(false);
    }
  };

  const deleteDocument = async (doc: LibraryDocument) => {
    if (!doc.deletable || deletingDocumentId) return;
    const confirmed = window.confirm(
      "Delete this source from the current index? Historical citations and reports are preserved, and already-running research may finish with its frozen evidence.",
    );
    if (!confirmed) return;
    setDeletingDocumentId(doc.id);
    setError(undefined);
    setSourceNotice("");
    try {
      const result = await api.deleteLibraryDocument(doc.id);
      await loadDocs();
      if (selectedDoc?.id === doc.id) {
        setSelectedDoc(undefined);
        setSourceSessions([]);
        setSourceSnapshots([]);
        setSnapshotPreview(undefined);
      }
      setSourceNotice(
        result.status === "deleted_with_cleanup_warning"
          ? "Source removed from the index; a local cleanup warning was recorded."
          : "Source removed from the current index. Historical citations were preserved.",
      );
    } catch (e) {
      if (e instanceof ApiError && e.code === "DOCUMENT_NOT_FOUND") {
        await loadDocs();
        if (selectedDoc?.id === doc.id) setSelectedDoc(undefined);
        setSourceNotice("The source was already removed.");
      } else {
        fail(e);
      }
    } finally {
      setDeletingDocumentId("");
    }
  };

  const addNote = async () => {
    const content = draft.trim();
    if (!content) return;
    setAddingNote(true);
    setError(undefined);
    try {
      await api.addLibraryNote({ content });
      setDraft("");
      await loadNotes();
    } catch (e) {
      fail(e);
    } finally {
      setAddingNote(false);
    }
  };

  const uploadReport = async () => {
    if (!uploadFile || uploadingReport) return;
    setUploadingReport(true);
    setError(undefined);
    try {
      await api.uploadLibraryReport(uploadFile, uploadTitle.trim());
      setUploadFile(null);
      setUploadTitle("");
      await loadReports();
    } catch (e) {
      fail(e);
    } finally {
      setUploadingReport(false);
    }
  };

  const selectSource = async (doc: LibraryDocument) => {
    setSelectedDoc(doc);
    setSnapshotPreview(undefined);
    setError(undefined);
    try {
      const [sessions, snapshots] = await Promise.all([
        api.listSourceSessions(doc.id),
        api.listSourceSnapshots(doc.id),
      ]);
      setSourceSessions(sessions.artifacts);
      setSourceSnapshots(snapshots.snapshots);
      const latest = snapshots.snapshots.find((snap) => snap.status === "ok");
      if (latest) {
        const preview = await api.getLibrarySnapshot(latest.id);
        setSnapshotPreview(preview.snapshot);
      }
    } catch (e) {
      fail(e);
    }
  };

  const selectReport = async (report: LibraryReport) => {
    setError(undefined);
    try {
      const res = await api.getLibraryReport(report.id);
      setSelectedReport(res.report);
    } catch (e) {
      fail(e);
    }
  };

  useEffect(() => {
    if (!documentId || !docsLoaded) return;
    if (deepLinkApplied.current.source === documentId) return;
    const doc = docs.find((item) => item.id === documentId);
    deepLinkApplied.current.source = documentId;
    if (!doc) {
      setSelectedDoc(undefined);
      setSourceSessions([]);
      setSourceSnapshots([]);
      setSnapshotPreview(undefined);
      setSourceNotice(
        "This historical source is no longer in the current index. Its saved citations and reports remain available.",
      );
      return;
    }
    void selectSource(doc);
  }, [documentId, docs, docsLoaded]);

  useEffect(() => {
    if (!reportId || reports.length === 0) return;
    if (deepLinkApplied.current.report === reportId) return;
    const report = reports.find((item) => item.id === reportId);
    if (!report) return;
    deepLinkApplied.current.report = reportId;
    void selectReport(report);
  }, [reportId, reports]);

  useEffect(() => {
    if (!noteId || tab !== "notes" || notes.length === 0) return;
    if (deepLinkApplied.current.note === noteId) return;
    const note = notes.find((item) => item.id === noteId);
    if (!note) return;
    deepLinkApplied.current.note = noteId;
    requestAnimationFrame(() => {
      document.querySelector(`[data-note-id="${noteId}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }, [noteId, tab, notes]);

  return (
    <div ref={ref} className="mx-auto w-full max-w-[920px] px-6 py-8">
      <h1 data-anim className="mb-4 text-2xl font-bold text-text">
        Library
      </h1>

      <div data-anim className="mb-5 flex gap-1 rounded-md border border-border bg-surface p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => switchTab(t.id)}
            aria-pressed={tab === t.id}
            className={cn(
              "flex-1 rounded-[6px] px-3 py-1.5 text-sm font-medium transition-colors focus-ring",
              tab === t.id ? "bg-accent-soft text-accent" : "text-text-muted hover:text-text-body",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {error && (
        <p role="alert" className="mb-3 text-sm text-danger">
          {error}
        </p>
      )}

      {tab === "sources" && (
        <div className="space-y-4">
          <div data-anim className="flex flex-col gap-2 sm:flex-row">
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addUrl()}
              placeholder="https://example.com/article"
              aria-label="URL to index"
              disabled={IS_DEMO}
            />
            <Button onClick={addUrl} disabled={IS_DEMO || busy || !url.trim()}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />} Index
            </Button>
          </div>
          <div data-anim className="rounded-md border border-border bg-surface p-3">
            <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,220px)_auto] sm:items-end">
              <label className="grid min-w-0 gap-1 text-xs font-medium text-text-muted">
                Local document
                <Input
                  aria-label="Local document file"
                  ref={documentInputRef}
                  type="file"
                  accept=".pdf,.docx,.xlsx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                  onChange={(event) => setDocumentFile(event.target.files?.[0] ?? null)}
                  disabled={IS_DEMO || importingDocument}
                />
              </label>
              <label className="grid min-w-0 gap-1 text-xs font-medium text-text-muted">
                Optional title
                <Input
                  aria-label="Local document title"
                  value={documentTitle}
                  onChange={(event) => setDocumentTitle(event.target.value)}
                  disabled={IS_DEMO || importingDocument}
                  placeholder="Keep the filename by default"
                />
              </label>
              <Button
                type="button"
                onClick={importDocument}
                disabled={IS_DEMO || importingDocument || !documentFile}
              >
                {importingDocument ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                Import
              </Button>
            </div>
            <p className="mt-2 text-xs text-text-faint">
              PDF, DOCX, or XLSX up to 10 MiB. Import is local and makes no Search, Extract, LLM,
              rerank, or embedding calls; the original binary is not retained.
            </p>
          </div>
          {sourceNotice && (
            <p role="status" className="text-sm text-text-muted">
              {sourceNotice}
            </p>
          )}
          {docs.length === 0 ? (
            <Card data-anim>
              <CardContent className="py-10 text-center text-sm text-text-muted">
                Index a URL to build your local, searchable corpus.
              </CardContent>
            </Card>
          ) : (
            <ul className="space-y-2">
              {docs.map((d) => (
                <li key={d.id} data-anim>
                  <Card className={cn(selectedDoc?.id === d.id && "ring-2 ring-accent")}>
                    <CardContent className="py-3">
                      <div className="flex min-w-0 items-center gap-2">
                        <button
                          type="button"
                          onClick={() => selectSource(d)}
                          className="flex min-w-0 flex-1 items-center gap-3 text-left focus-ring"
                        >
                          {isLocalDocument(d) ? (
                            <FileText className="h-4 w-4 shrink-0 text-text-muted" aria-hidden />
                          ) : (
                            <Globe className="h-4 w-4 shrink-0 text-text-muted" aria-hidden />
                          )}
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-medium text-text">
                              {d.title || d.filename || domainOf(d.url || "")}
                            </span>
                            <span className="block truncate text-xs text-text-faint">
                              {isLocalDocument(d)
                                ? `${d.filename || "Local document"} · ${d.content_type.toUpperCase()}`
                                : d.url}
                            </span>
                          </span>
                          <span className="shrink-0 text-xs text-text-muted">{d.chunks} chunks</span>
                        </button>
                        {d.deletable && (
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            aria-label={`Delete ${d.title || d.filename || "source"}`}
                            onClick={() => deleteDocument(d)}
                            disabled={IS_DEMO || Boolean(deletingDocumentId)}
                            className="shrink-0 text-danger"
                          >
                            {deletingDocumentId === d.id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Trash2 className="h-4 w-4" />
                            )}
                          </Button>
                        )}
                      </div>
                      {selectedDoc?.id === d.id && (
                        <div className="mt-3 border-t border-border pt-3 text-sm">
                          <div className="mb-2 flex flex-wrap items-center gap-3 text-xs text-text-muted">
                            {isExternalSource(d) && d.url ? (
                              <a className="inline-flex items-center gap-1 text-accent" href={d.url}>
                                <ExternalLink className="h-3.5 w-3.5" /> Open latest web page
                              </a>
                            ) : (
                              <span>Local evidence document</span>
                            )}
                            <span>{sourceSnapshots.length} snapshots</span>
                            <span>{sourceSessions.length} linked runs</span>
                          </div>
                          {snapshotPreview?.text ? (
                            <pre className="max-h-56 overflow-auto whitespace-pre-wrap rounded-md bg-bg-subtle p-3 text-xs text-text-body">
                              {snapshotPreview.text}
                            </pre>
                          ) : isLocalDocument(d) ? (
                            <p className="text-xs text-text-muted">
                              Extracted text is searchable as local evidence; the original file is not retained.
                            </p>
                          ) : (
                            <p className="text-xs text-text-muted">No local snapshot preview yet.</p>
                          )}
                          {sourceSessions.length > 0 && (
                            <div className="mt-3 flex flex-wrap gap-2">
                              {sourceSessions.map((item, index) => (
                                <a
                                  key={`${item.session_id}-${item.run_id}-${index}`}
                                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-text-body hover:bg-bg-subtle"
                                  href={item.run_id ? `/run/${item.run_id}` : `/session/${item.session_id}`}
                                >
                                  <History className="h-3.5 w-3.5" />
                                  {item.run_id || item.session_id}
                                </a>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {tab === "notes" && (
        <div className="space-y-4">
          <div data-anim className="space-y-2">
            <Textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Jot down a research note..."
              aria-label="New note"
              rows={3}
              disabled={IS_DEMO}
            />
            <Button onClick={addNote} disabled={IS_DEMO || addingNote || !draft.trim()}>
              {addingNote ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />} Add note
            </Button>
          </div>
          {notes.length === 0 ? (
            <Card data-anim>
              <CardContent className="py-10 text-center text-sm text-text-muted">
                Saved notes from your research will appear here.
              </CardContent>
            </Card>
          ) : (
            <ul className="space-y-2">
              {notes.map((n) => (
                <li key={n.id} data-anim data-note-id={n.id}>
                  <Card className={cn(noteId === n.id && "ring-2 ring-accent")}>
                    <CardContent className="flex items-start gap-3 py-3">
                      <StickyNote className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" aria-hidden />
                      <div className="min-w-0 flex-1">
                        <div className="mb-1 flex items-center gap-2">
                          <span className="rounded-md bg-bg-subtle px-2 py-0.5 text-xs text-text-muted">
                            {noteTag(n)}
                          </span>
                          {n.created_at && <span className="text-xs text-text-faint">{formatTime(n.created_at)}</span>}
                          {artifactHref(n) && (
                            <a
                              className="inline-flex items-center gap-1 text-xs text-accent"
                              href={artifactHref(n)}
                            >
                              <History className="h-3.5 w-3.5" />
                              {n.run_id || n.session_id}
                            </a>
                          )}
                        </div>
                        <p className="whitespace-pre-wrap text-sm text-text-body">{n.content}</p>
                      </div>
                    </CardContent>
                  </Card>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {tab === "reports" && (
        <div className="grid gap-4 md:grid-cols-[minmax(0,280px)_1fr]">
          <div className="space-y-2">
            <div className="rounded-md border border-border bg-surface p-3">
              <div className="space-y-2">
                <Input
                  aria-label="Report title"
                  value={uploadTitle}
                  onChange={(e) => setUploadTitle(e.target.value)}
                  placeholder="Report title"
                  disabled={IS_DEMO}
                />
                <Input
                  aria-label="Report file"
                  type="file"
                  accept=".md,.txt,text/markdown,text/plain"
                  onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                  disabled={IS_DEMO}
                />
                <Button type="button" size="sm" onClick={uploadReport} disabled={IS_DEMO || uploadingReport || !uploadFile}>
                  {uploadingReport ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                  Upload
                </Button>
              </div>
            </div>
            {reports.length === 0 ? (
              <Card data-anim>
                <CardContent className="py-10 text-center text-sm text-text-muted">
                  Deep and Council reports will appear here.
                </CardContent>
              </Card>
            ) : (
              reports.map((report) => (
                <button
                  key={report.id}
                  type="button"
                  onClick={() => selectReport(report)}
                  className={cn(
                    "w-full rounded-md border border-border bg-surface p-3 text-left transition-colors focus-ring",
                    selectedReport?.id === report.id && "border-accent bg-accent-soft",
                  )}
                >
                  <span className="mb-1 flex items-center gap-2 text-sm font-medium text-text">
                    <FileText className="h-4 w-4 shrink-0 text-text-muted" />
                    <span className="truncate">{report.title}</span>
                  </span>
                  <span className="text-xs text-text-muted">
                    {report.mode} {report.created_at ? `- ${formatTime(report.created_at)}` : ""}
                  </span>
                </button>
              ))
            )}
          </div>
          <div className="min-h-[240px] rounded-md border border-border bg-surface p-4">
            {selectedReport ? (
              <div>
                <div className="mb-3">
                  <h2 className="text-lg font-semibold text-text">{selectedReport.title}</h2>
                  <p className="text-xs text-text-muted">
                    {selectedReport.mode}
                    {artifactHref(selectedReport) ? (
                      <>
                        {" - "}
                        <a className="inline-flex items-center gap-1 text-accent" href={artifactHref(selectedReport)}>
                          <History className="h-3.5 w-3.5" />
                          {selectedReport.run_id || selectedReport.session_id}
                        </a>
                      </>
                    ) : null}
                  </p>
                </div>
                <AnswerBody text={selectedReport.body_md || selectedReport.body_preview || ""} citations={[]} />
                {selectedReport.children && selectedReport.children.length > 0 && (
                  <div className="mt-5 space-y-2 border-t border-border pt-3">
                    {selectedReport.children.map((child) => (
                      <details key={child.id} className="rounded-md bg-bg-subtle p-3">
                        <summary className="cursor-pointer text-sm font-medium text-text">{child.title}</summary>
                        <p className="mt-2 whitespace-pre-wrap text-sm text-text-body">{child.body_preview}</p>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-text-muted">
                Select a report to preview it.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
