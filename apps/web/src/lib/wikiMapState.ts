type WikiStateKind = "expanded" | "focus" | "cross-links";

function safeLocalStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

export function wikiStateStorageKey(kind: WikiStateKind, workspaceId: string, snapshotId: string): string {
  return `kg-wiki-${kind}:${workspaceId}:${snapshotId}`;
}

export function readWikiStringSet(kind: "expanded", workspaceId: string, snapshotId: string): Set<string> {
  const storage = safeLocalStorage();
  if (!storage) return new Set();
  try {
    const raw = storage.getItem(wikiStateStorageKey(kind, workspaceId, snapshotId));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? new Set(parsed.filter((item): item is string => typeof item === "string")) : new Set();
  } catch {
    return new Set();
  }
}

export function writeWikiStringSet(kind: "expanded", workspaceId: string, snapshotId: string, values: Set<string>): void {
  const storage = safeLocalStorage();
  if (!storage) return;
  storage.setItem(wikiStateStorageKey(kind, workspaceId, snapshotId), JSON.stringify([...values].sort()));
}

export function readWikiString(kind: "focus", workspaceId: string, snapshotId: string): string | undefined {
  const storage = safeLocalStorage();
  if (!storage) return undefined;
  const raw = storage.getItem(wikiStateStorageKey(kind, workspaceId, snapshotId));
  return raw?.trim() || undefined;
}

export function writeWikiString(kind: "focus", workspaceId: string, snapshotId: string, value?: string): void {
  const storage = safeLocalStorage();
  if (!storage) return;
  const key = wikiStateStorageKey(kind, workspaceId, snapshotId);
  if (!value) storage.removeItem(key);
  else storage.setItem(key, value);
}

export function readWikiBoolean(kind: "cross-links", workspaceId: string, snapshotId: string): boolean {
  const storage = safeLocalStorage();
  return storage?.getItem(wikiStateStorageKey(kind, workspaceId, snapshotId)) === "true";
}

export function writeWikiBoolean(kind: "cross-links", workspaceId: string, snapshotId: string, value: boolean): void {
  const storage = safeLocalStorage();
  if (!storage) return;
  storage.setItem(wikiStateStorageKey(kind, workspaceId, snapshotId), value ? "true" : "false");
}

export function clearWikiSnapshotState(workspaceId: string, snapshotId: string): void {
  const storage = safeLocalStorage();
  if (!storage) return;
  (["expanded", "focus", "cross-links"] as const).forEach((kind) => storage.removeItem(wikiStateStorageKey(kind, workspaceId, snapshotId)));
}
