import { stringAttr } from "@/components/graph/GraphNode";
import {
  libraryNotePath,
  libraryPathForTab,
  libraryReportPath,
  librarySourcePath,
  normalizeLibraryJumpUri,
} from "@/lib/libraryRoutes";
import type { KnowledgeGraphNode, KnowledgeGraphWikiEvidenceRef } from "@/lib/types";

function evidenceEntityId(ref: KnowledgeGraphWikiEvidenceRef, prefix: string): string {
  const value = String(ref.id || ref.node_key || "").trim();
  return value.startsWith(`${prefix}:`) ? value.slice(prefix.length + 1) : value;
}

/** Resolve bounded Wiki evidence references to existing in-app surfaces. */
export function resolveWikiEvidenceJumpTarget(
  ref: KnowledgeGraphWikiEvidenceRef,
  context: { sessionId?: string } = {},
): string | null {
  const type = String(ref.type || "").trim().toLowerCase();
  if (type === "source") {
    const id = evidenceEntityId(ref, "source");
    return id ? librarySourcePath(id) : libraryPathForTab("sources");
  }
  if (type === "note") {
    const id = evidenceEntityId(ref, "note");
    return id ? libraryNotePath(id) : libraryPathForTab("notes");
  }
  if (type === "report") {
    const id = evidenceEntityId(ref, "report");
    return id ? libraryReportPath(id) : libraryPathForTab("reports");
  }
  if (type === "message") return context.sessionId ? `/session/${context.sessionId}` : null;
  if (type === "memory") {
    const id = evidenceEntityId(ref, "memory");
    return id ? `/memory/${encodeURIComponent(id)}` : "/memory";
  }
  return null;
}

/** Resolve the in-app navigation target for a knowledge-graph node. */
export function resolveGraphJumpTarget(node: KnowledgeGraphNode): string {
  const jumpUri = stringAttr(node, "jump_uri");
  if (/^\/(run|session)\//.test(jumpUri)) return jumpUri;

  const libraryJump = normalizeLibraryJumpUri(jumpUri);
  if (libraryJump) return libraryJump;

  if (node.type === "session") {
    const sessionId = stringAttr(node, "session_id");
    return sessionId ? `/session/${sessionId}` : "/";
  }
  if (node.type === "memory") {
    const memoryId = stringAttr(node, "memory_id");
    return memoryId ? `/memory/${encodeURIComponent(memoryId)}` : "/memory";
  }
  if (node.type === "note") {
    const noteId = stringAttr(node, "note_id");
    return noteId ? libraryNotePath(noteId) : libraryPathForTab("notes");
  }
  if (node.type === "report") {
    const reportId = stringAttr(node, "report_id");
    return reportId ? libraryReportPath(reportId) : libraryPathForTab("reports");
  }
  if (node.type === "source") {
    const documentId = stringAttr(node, "document_id");
    return documentId ? librarySourcePath(documentId) : "/library";
  }
  return "/";
}
