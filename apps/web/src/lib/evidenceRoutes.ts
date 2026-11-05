import type { EvidenceSource } from "./types";

export function evidenceKey(source: EvidenceSource): string {
  const origin = source.origin ?? "external";
  const artifactType = source.artifact_type ?? "web_document";
  if (source.artifact_id && origin !== "external") return `${origin}:${artifactType}:${source.artifact_id}`;
  if (source.provenance_refs?.length) return source.provenance_refs[0];
  return `external:web_document:${source.document_id || source.url || source.chunk_id}`;
}

export function evidenceHref(source: EvidenceSource): { href: string; external: boolean } | null {
  const artifactId = source.artifact_id?.trim();
  if (artifactId && source.artifact_type === "library_note") {
    return { href: `/library/notes/${encodeURIComponent(artifactId)}`, external: false };
  }
  if (artifactId && source.artifact_type === "library_report") {
    return { href: `/library/reports/${encodeURIComponent(artifactId)}`, external: false };
  }
  if (artifactId && source.artifact_type === "library_document") {
    return { href: `/library/sources/${encodeURIComponent(artifactId)}`, external: false };
  }
  if (artifactId && source.artifact_type === "memory") {
    return { href: `/memory/${encodeURIComponent(artifactId)}`, external: false };
  }
  return source.url ? { href: source.url, external: true } : null;
}

export function evidenceArtifactLabel(source: EvidenceSource): string {
  if (source.artifact_type === "library_note") return "note";
  if (source.artifact_type === "library_report") return "report";
  if (source.artifact_type === "library_document") return "document";
  if (source.artifact_type === "memory") return "memory";
  return "web";
}
