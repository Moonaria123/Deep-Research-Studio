export type LibraryTab = "sources" | "notes" | "reports";

/** Default Library path for a tab (no item selection). */
export function libraryPathForTab(tab: LibraryTab): string {
  if (tab === "sources") return "/library";
  return `/library/${tab}`;
}

export function libraryNotePath(noteId: string): string {
  return `/library/notes/${encodeURIComponent(noteId)}`;
}

export function libraryReportPath(reportId: string): string {
  return `/library/reports/${encodeURIComponent(reportId)}`;
}

export function librarySourcePath(documentId: string): string {
  return `/library/sources/${encodeURIComponent(documentId)}`;
}

/** Resolve active Library tab from the current pathname. */
export function libraryTabFromPath(pathname: string): LibraryTab {
  if (pathname.includes("/library/notes")) return "notes";
  if (pathname.includes("/library/reports")) return "reports";
  return "sources";
}

const LIBRARY_JUMP_RE = /^\/library\/(notes|reports|sources)\/([^/?#]+)$/;

/** Returns a canonical in-app Library deep link when the URI matches `/library/{tab}/{id}`. */
export function normalizeLibraryJumpUri(uri: string): string | null {
  const trimmed = uri.trim();
  const match = trimmed.match(LIBRARY_JUMP_RE);
  if (!match) return null;
  const [, kind, rawId] = match;
  const id = decodeURIComponent(rawId);
  if (kind === "notes") return libraryNotePath(id);
  if (kind === "reports") return libraryReportPath(id);
  return librarySourcePath(id);
}
