function safeLocalStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

export function sessionMapCrossLinksStorageKey(workspaceId: string, graphVersionId: string): string {
  return `kg-session-cross-links:${workspaceId}:${graphVersionId}`;
}

export function readSessionMapCrossLinks(workspaceId: string, graphVersionId: string): boolean {
  const storage = safeLocalStorage();
  return storage?.getItem(sessionMapCrossLinksStorageKey(workspaceId, graphVersionId)) === "true";
}

export function writeSessionMapCrossLinks(workspaceId: string, graphVersionId: string, value: boolean): void {
  const storage = safeLocalStorage();
  if (!storage) return;
  storage.setItem(sessionMapCrossLinksStorageKey(workspaceId, graphVersionId), value ? "true" : "false");
}

export function clearSessionMapState(workspaceId: string, graphVersionId: string): void {
  const storage = safeLocalStorage();
  if (!storage) return;
  storage.removeItem(sessionMapCrossLinksStorageKey(workspaceId, graphVersionId));
}
